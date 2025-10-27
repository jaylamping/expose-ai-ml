"""
FastAPI endpoint for multi-stage bot detection pipeline.
"""
import time
import logging
import traceback
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
from utils.type_conversion import convert_numpy_types

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bot_detection.log')
    ]
)
logger = logging.getLogger(__name__)


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

class AnalyzeUserRequest(BaseModel):
    """Request model for bot analysis."""
    user_id: str = Field(..., description="Reddit username or user ID")
    comments: List[UserComment] = Field(..., description="List of user comments to analyze")
    options: Optional[AnalysisOptions] = Field(default_factory=AnalysisOptions, description="Analysis options")

class AnalyzeUserResponse(BaseModel):
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
        
        @self.app.post("/api/v1/analyze/user", response_model=AnalyzeUserResponse)
        async def analyze_user(request: AnalyzeUserRequest):
            """Analyze a user's comments for bot detection."""
            logger.info(f"Starting analysis for user: {request.user_id}")
            logger.debug(f"Request details - Comments: {len(request.comments)}, Options: {request.options}")
            
            try:
                # Initialize models if not done yet
                if not self._initialized:
                    logger.info("Models not initialized, starting initialization...")
                    await self._initialize_models()
                    logger.info("Model initialization completed")
                else:
                    logger.debug("Models already initialized")
                
                # Validate request
                logger.debug("Validating request...")
                if not request.comments:
                    logger.warning("No comments provided in request")
                    raise HTTPException(status_code=400, detail="No comments provided")
                
                if len(request.comments) > settings.max_comments_per_request:
                    logger.warning(f"Too many comments: {len(request.comments)} > {settings.max_comments_per_request}")
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Too many comments. Maximum allowed: {settings.max_comments_per_request}"
                    )
                
                logger.debug(f"Request validation passed - {len(request.comments)} comments to analyze")
                
                # Run analysis
                logger.info("Starting user comment analysis...")
                result = await self._analyze_user_comments(request)
                logger.info(f"Analysis completed successfully for user: {request.user_id}")
                
                return result
                
            except HTTPException as e:
                logger.error(f"HTTP Exception in analyze_user: {e.status_code} - {e.detail}")
                raise e
            except Exception as e:
                logger.error(f"Unexpected error in analyze_user: {str(e)}")
                logger.error(f"Full traceback: {traceback.format_exc()}")
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
        logger.info("Starting model initialization process...")
        try:
            logger.info("Initializing bot detection models...")
            
            # Initialize preprocessor
            logger.debug("Initializing preprocessor...")
            self.preprocessor = RedditPreprocessor(max_length=settings.max_sequence_length)
            logger.info("Preprocessor initialized successfully")
            
            # Initialize fast detector
            logger.debug("Initializing fast detector...")
            self.fast_detector = FastDetector()
            logger.info("Fast detector initialized successfully")
            
            # Initialize deep detector
            logger.debug("Initializing deep detector...")
            self.deep_detector = DeepDetector()
            logger.info("Deep detector initialized successfully")
            
            # Initialize statistical analyzer
            logger.debug("Initializing statistical analyzer...")
            self.statistical_analyzer = StatisticalAnalyzer()
            logger.info("Statistical analyzer initialized successfully")
            
            # Initialize ensemble scorer
            logger.debug("Initializing ensemble scorer...")
            self.ensemble_scorer = EnsembleScorer()
            logger.info("Ensemble scorer initialized successfully")
            
            self._initialized = True
            logger.info("All models initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {str(e)}")
            logger.error(f"Model initialization traceback: {traceback.format_exc()}")
            raise e
    
    async def _analyze_user_comments(self, request: AnalyzeUserRequest) -> AnalyzeUserResponse:
        """Analyze user comments using the multi-stage pipeline."""
        start_time = time.time()
        logger.info(f"Starting multi-stage analysis for user: {request.user_id}")
        
        try:
            # Stage 1: Fast screening
            logger.info(f"Stage 1: Fast screening for user {request.user_id}")
            fast_result = await self._run_fast_analysis(request)
            logger.info(f"Fast screening completed - Bot score: {fast_result.bot_score:.3f}, Confidence: {fast_result.confidence:.3f}")
            
            # Check if we should skip deeper analysis
            should_skip_deep = (
                self.ensemble_scorer.should_skip_deep_analysis(fast_result) and 
                not request.options.force_full_analysis
            )
            
            if request.options.fast_only or should_skip_deep:
                logger.info(f"Skipping deep analysis (fast_only={request.options.fast_only}, should_skip={should_skip_deep})")
                
                # Return fast-only result
                result = self.ensemble_scorer.combine_fast_and_deep(fast_result)
                result.update({
                    "user_id": request.user_id,
                    "comments_analyzed": fast_result.breakdown.get("comments_analyzed", 0) if fast_result.breakdown else 0,
                    "total_comments": len(request.comments),
                    "explanation": self.ensemble_scorer.get_confidence_explanation(result)
                })
                
                total_time = (time.time() - start_time) * 1000
                logger.info(f"Fast-only analysis complete for user {request.user_id} in {total_time:.1f}ms")
                return AnalyzeUserResponse(**result)
            
            # Stage 2: Deep analysis
            logger.info(f"Stage 2: Deep analysis for user {request.user_id}")
            deep_result = await self._run_deep_analysis(request)
            logger.info(f"Deep analysis completed - Bot score: {deep_result.bot_score:.3f}, Confidence: {deep_result.confidence:.3f}")
            
            # Stage 3: Statistical analysis
            logger.info(f"Stage 3: Statistical analysis for user {request.user_id}")
            statistical_results = await self._run_statistical_analysis(request)
            logger.info(f"Statistical analysis completed - Processing time: {statistical_results.get('processing_time_ms', 0):.1f}ms")
            
            # Stage 4: Ensemble scoring
            logger.info(f"Stage 4: Ensemble scoring for user {request.user_id}")
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
            logger.info(f"Full analysis complete for user {request.user_id} in {total_time:.1f}ms - Final bot score: {result.get('bot_score', 0):.3f}")
            
            # Convert numpy types to native Python types for JSON serialization
            return convert_numpy_types(result)
            
        except Exception as e:
            logger.error(f"Error in _analyze_user_comments for user {request.user_id}: {str(e)}")
            logger.error(f"Analysis error traceback: {traceback.format_exc()}")
            raise e
    
    async def _run_fast_analysis(self, request: AnalyzeUserRequest) -> AnalysisResult:
        """Run fast screening analysis."""
        start_time = time.time()
        logger.debug(f"Starting fast analysis for {len(request.comments)} comments")
        
        try:
            # Analyze comments
            if request.options.use_context and any(comment.parent_comment for comment in request.comments):
                logger.debug("Using context for fast analysis")
                # Extract parent comments for context
                parent_contexts = [comment.parent_comment for comment in request.comments if comment.parent_comment]
                formatted_pairs = self.preprocessor.create_context_pairs(
                    [comment.comment for comment in request.comments], 
                    [{"parent": ctx.comment} for ctx in parent_contexts]
                )
                analysis_result = self.fast_detector.analyze_user_comments(formatted_pairs)
            else:
                logger.debug("Running fast analysis without context")
                analysis_result = self.fast_detector.analyze_user_comments([comment.comment for comment in request.comments])
            
            processing_time = (time.time() - start_time) * 1000
            logger.debug(f"Fast analysis completed in {processing_time:.1f}ms")
            
            return AnalysisResult(
                stage="fast_screening",
                bot_score=analysis_result["bot_score"],
                confidence=analysis_result["confidence"],
                processing_time_ms=processing_time,
                breakdown=analysis_result,
                should_skip_next=analysis_result.get("should_skip_deep", False)
            )
            
        except Exception as e:
            logger.error(f"Error in fast analysis: {str(e)}")
            logger.error(f"Fast analysis traceback: {traceback.format_exc()}")
            raise e
    
    async def _run_deep_analysis(self, request: AnalyzeUserRequest) -> AnalysisResult:
        """Run deep analysis."""
        start_time = time.time()
        logger.debug(f"Starting deep analysis for {len(request.comments)} comments")
        
        try:
            # Analyze comments
            if request.options.use_context and any(comment.parent_comment for comment in request.comments):
                logger.debug("Using context for deep analysis")
                # Extract parent comments for context
                parent_contexts = [comment.parent_comment for comment in request.comments if comment.parent_comment]
                analysis_result = self.deep_detector.analyze_with_context(
                    [comment.comment for comment in request.comments], 
                    [{"parent": ctx.comment} for ctx in parent_contexts]
                )
            else:
                logger.debug("Running deep analysis without context")
                analysis_result = self.deep_detector.analyze_user_comments([comment.comment for comment in request.comments])
            
            processing_time = (time.time() - start_time) * 1000
            logger.debug(f"Deep analysis completed in {processing_time:.1f}ms")
            
            return AnalysisResult(
                stage="deep_analysis",
                bot_score=analysis_result["bot_score"],
                confidence=analysis_result["confidence"],
                processing_time_ms=processing_time,
                breakdown=analysis_result
            )
            
        except Exception as e:
            logger.error(f"Error in deep analysis: {str(e)}")
            logger.error(f"Deep analysis traceback: {traceback.format_exc()}")
            raise e
    
    async def _run_statistical_analysis(self, request: AnalyzeUserRequest) -> AnalysisResult:
        """Run statistical analysis."""
        start_time = time.time()
        logger.debug(f"Starting statistical analysis for {len(request.comments)} comments")
        
        try:
            # Clean comments
            logger.debug("Preprocessing comments for statistical analysis...")
            comment_texts = [comment.comment for comment in request.comments]
            preprocessed = self.preprocessor.batch_preprocess(comment_texts, include_features=True)
            cleaned_comments = preprocessed['cleaned_comments']
            logger.debug(f"Preprocessed {len(cleaned_comments)} comments")
            
            # Run all statistical analyses
            logger.debug("Running statistical analyzers...")
            statistical_results = self.statistical_analyzer.analyze_all_signals(cleaned_comments)
            
            processing_time = (time.time() - start_time) * 1000
            statistical_results["processing_time_ms"] = processing_time
            logger.debug(f"Statistical analysis completed in {processing_time:.1f}ms")
            
            return statistical_results
            
        except Exception as e:
            logger.error(f"Error in statistical analysis: {str(e)}")
            logger.error(f"Statistical analysis traceback: {traceback.format_exc()}")
            raise e
    
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
