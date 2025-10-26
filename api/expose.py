"""
FastAPI endpoint for multi-stage bot detection pipeline.
"""
import time
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from config.settings import settings
from models.detectors.fast_detector import FastDetector
from models.detectors.deep_detector import DeepDetector
from models.detectors.statistical_analyzer import StatisticalAnalyzer
from models.ensemble import EnsembleScorer, AnalysisResult
from utils.preprocessing import RedditPreprocessor


# Pydantic models for API
class AnalysisOptions(BaseModel):
    """Options for analysis."""
    fast_only: bool = False
    include_breakdown: bool = True
    use_context: bool = True
    force_full_analysis: bool = False

class BaseComment(BaseModel):
    """Base comment model."""
    comment_id: str = Field(..., description="Comment ID")
    comment: str = Field(..., description="Comment text")
    created_at: Optional[str] = Field(None, description="Comment creation timestamp")
    updated_at: Optional[str] = Field(None, description="Comment update timestamp")

class UserComment(BaseComment):
    """User comment model."""
    parent_comment: Optional[BaseComment] = Field(None, description="Parent comment")
    child_comment: Optional[BaseComment] = Field(None, description="Child comment")

class UserCommentRequest(BaseModel):
    """Request model for bot analysis."""
    user_id: str = Field(..., description="Reddit username or user ID")
    comments: List[UserComment] = Field(..., description="List of user comments to analyze")
    options: Optional[AnalysisOptions] = Field(default_factory=AnalysisOptions, description="Analysis options")


class UserCommentResponse(BaseModel):
    """Response model for bot analysis."""
    user_id: str
    bot_score: float = Field(..., description="Bot probability score (0-100)")
    confidence: float = Field(..., description="Confidence in the analysis (0-100)")
    is_likely_bot: bool = Field(..., description="Whether the user is likely a bot")
    stage: str = Field(..., description="Analysis stage used")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    comments_analyzed: int = Field(..., description="Number of comments analyzed")
    total_comments: int = Field(..., description="Total number of comments provided")
    breakdown: Optional[Dict[str, Any]] = Field(None, description="Detailed breakdown of scores")
    explanation: Optional[str] = Field(None, description="Human-readable explanation")


class ExposeAPI:
    """Main API class for bot detection."""
    
    def __init__(self):
        """Initialize the API with all detectors."""
        self.app = FastAPI(
            title="Bot Detection API",
            description="Multi-stage bot detection for social media content",
            version="1.0.0"
        )
        
        # Initialize components
        self.fast_detector = None
        self.deep_detector = None
        self.statistical_analyzer = None
        self.ensemble_scorer = None
        self.preprocessor = None
        
        # Setup routes
        self._setup_routes()
        
        # Initialize models (lazy loading)
        self._initialized = False
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/")
        async def root():
            return {"message": "Bot Detection API", "version": "1.0.0"}
        
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "initialized": self._initialized}
        
        @self.app.post("/api/v1/analyze-user-comments", response_model=UserCommentResponse)
        async def analyze_user_comments(request: UserCommentRequest):
            """Analyze a user's comments for bot detection."""
            try:
                # Initialize models if not done yet
                if not self._initialized:
                    await self._initialize_models()
                
                # Validate request
                if not request.comments:
                    raise HTTPException(status_code=400, detail="No comments provided")
                
                if len(request.comments) > settings.max_comments_per_request:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Too many comments. Maximum allowed: {settings.max_comments_per_request}"
                    )
                
                # Run analysis
                result = await self._analyze_user_comments(request)
                
                return UserCommentResponse(**result)
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
        
        @self.app.get("/api/v1/models/info")
        async def get_models_info():
            """Get information about loaded models."""
            if not self._initialized:
                return {"error": "Models not initialized"}
            
            return {
                "fast_detector": self.fast_detector.get_model_info() if self.fast_detector else None,
                "deep_detector": self.deep_detector.get_model_info() if self.deep_detector else None,
                "statistical_analyzer": {
                    "available_analyzers": self.statistical_analyzer.get_available_analyzers() if self.statistical_analyzer else []
                },
                "ensemble_scorer": self.ensemble_scorer.get_ensemble_info() if self.ensemble_scorer else None
            }
    
    async def _initialize_models(self):
        """Initialize all models (lazy loading)."""
        try:
            print("Initializing bot detection models...")
            
            # Initialize preprocessor
            self.preprocessor = RedditPreprocessor(max_length=settings.max_sequence_length)
            print("Preprocessor initialized")
            
            # Initialize fast detector
            self.fast_detector = FastDetector()
            print("Fast detector initialized")
            
            # Initialize deep detector
            self.deep_detector = DeepDetector()
            print("Deep detector initialized")
            
            # Initialize statistical analyzer
            self.statistical_analyzer = StatisticalAnalyzer()
            print("Statistical analyzer initialized")
            
            # Initialize ensemble scorer
            self.ensemble_scorer = EnsembleScorer()
            print("Ensemble scorer initialized")
            
            self._initialized = True
            print("All models initialized successfully!")
            
        except Exception as e:
            print(f"Failed to initialize models: {e}")
            raise e
    
    async def _analyze_user_comments(self, request: UserCommentRequest) -> UserCommentResponse:
        """Analyze user comments using the multi-stage pipeline."""
        start_time = time.time()
        
        # Stage 1: Fast screening
        print(f"Stage 1: Fast screening for user {request.user_id}")
        fast_result = await self._run_fast_analysis(request)
        
        # Check if we should skip deeper analysis
        should_skip_deep = (
            self.ensemble_scorer.should_skip_deep_analysis(fast_result) and 
            not request.options.force_full_analysis
        )
        
        if request.options.fast_only or should_skip_deep:
            print(f"Skipping deep analysis (fast_only={request.options.fast_only}, should_skip={should_skip_deep})")
            
            # Return fast-only result
            result = self.ensemble_scorer.combine_fast_and_deep(fast_result)
            result.update({
                "user_id": request.user_id,
                "comments_analyzed": fast_result.breakdown.get("comments_analyzed", 0) if fast_result.breakdown else 0,
                "total_comments": len(request.comments),
                "explanation": self.ensemble_scorer.get_confidence_explanation(result)
            })
            
            return result
        
        # Stage 2: Deep analysis
        print(f"Stage 2: Deep analysis for user {request.user_id}")
        deep_result = await self._run_deep_analysis(request)
        
        # Stage 3: Statistical analysis
        print(f"Stage 3: Statistical analysis for user {request.user_id}")
        statistical_results = await self._run_statistical_analysis(request)
        
        # Stage 4: Ensemble scoring
        print(f"Stage 4: Ensemble scoring for user {request.user_id}")
        result = self.ensemble_scorer.combine_all_signals(
            fast_result, deep_result, statistical_results
        )
        
        # Add metadata
        result.update({
            "user_id": request.user_id,
            "comments_analyzed": deep_result.breakdown.get("comments_analyzed", 0) if deep_result.breakdown else 0,
            "total_comments": len(request.comments),
            "explanation": self.ensemble_scorer.get_confidence_explanation(result)
        })
        
        total_time = (time.time() - start_time) * 1000
        print(f"Analysis complete for user {request.user_id} in {total_time:.1f}ms")
        
        return result
    
    async def _run_fast_analysis(self, request: UserCommentRequest) -> AnalysisResult:
        """Run fast screening analysis."""
        start_time = time.time()
        
        # Analyze comments
        if request.options.use_context and request.parent_contexts:
            # Use context if available
            formatted_pairs = self.preprocessor.create_context_pairs(
                request.comments, 
                [{"parent": ctx.parent} for ctx in request.parent_contexts]
            )
            analysis_result = self.fast_detector.analyze_user_comments(formatted_pairs)
        else:
            analysis_result = self.fast_detector.analyze_user_comments(request.comments)
        
        processing_time = (time.time() - start_time) * 1000
        
        return AnalysisResult(
            stage="fast_screening",
            bot_score=analysis_result["bot_score"],
            confidence=analysis_result["confidence"],
            processing_time_ms=processing_time,
            breakdown=analysis_result,
            should_skip_next=analysis_result.get("should_skip_deep", False)
        )
    
    async def _run_deep_analysis(self, request: UserCommentRequest) -> AnalysisResult:
        """Run deep analysis."""
        start_time = time.time()
        
        # Analyze comments
        if request.options.use_context and request.parent_contexts:
            # Use context if available
            parent_contexts = [{"parent": ctx.parent} for ctx in request.parent_contexts]
            analysis_result = self.deep_detector.analyze_with_context(
                request.comments, parent_contexts
            )
        else:
            analysis_result = self.deep_detector.analyze_user_comments(request.comments)
        
        processing_time = (time.time() - start_time) * 1000
        
        return AnalysisResult(
            stage="deep_analysis",
            bot_score=analysis_result["bot_score"],
            confidence=analysis_result["confidence"],
            processing_time_ms=processing_time,
            breakdown=analysis_result
        )
    
    async def _run_statistical_analysis(self, request: UserCommentRequest) -> Dict[str, Any]:
        """Run statistical analysis."""
        start_time = time.time()
        
        # Clean comments
        preprocessed = self.preprocessor.batch_preprocess(request.comments, include_features=True)
        cleaned_comments = preprocessed['cleaned_comments']
        
        # Run all statistical analyses
        statistical_results = self.statistical_analyzer.analyze_all_signals(cleaned_comments)
        
        processing_time = (time.time() - start_time) * 1000
        statistical_results["processing_time_ms"] = processing_time
        
        return statistical_results
    
    def run(self, host: str = None, port: int = None, debug: bool = None):
        """Run the API server."""
        host = host or settings.api_host
        port = port or settings.api_port
        debug = debug or settings.api_debug
        
        print(f"Starting Bot Detection API on {host}:{port}")
        print(f"Debug mode: {debug}")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="debug" if debug else "info"
        )


# Global API instance
api = ExposeAPI()

# Export the FastAPI app for external use
app = api.app

if __name__ == "__main__":
    api.run()
