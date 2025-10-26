"""
LangChain integration with Ollama for local LLM applications.
"""
from typing import Dict, Any, Optional, List, Union
from langchain.llms import Ollama
from langchain.embeddings import OllamaEmbeddings
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain as LangChainLLMChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler

from config.settings import settings


class LLMChain:
    """
    Wrapper around LangChain for easy LLM integration.
    """
    
    def __init__(self, 
                 model_name: Optional[str] = None,
                 temperature: float = 0.7,
                 num_predict: Optional[int] = None,
                 **kwargs):
        """
        Initialize LLM chain with Ollama.
        
        Args:
            model_name: Name of the Ollama model to use
            temperature: Sampling temperature
            num_predict: Maximum tokens to generate
            **kwargs: Additional arguments for the LLM
        """
        self.model_name = model_name or settings.default_llm_model
        self.temperature = temperature
        self.num_predict = num_predict
        
        self.llm = self._create_llm(**kwargs)
        self.memory = ConversationBufferMemory()
        self.chains: Dict[str, LangChainLLMChain] = {}
    
    def _create_llm(self, **kwargs) -> LLM:
        """Create the Ollama LLM instance."""
        return Ollama(
            model=self.model_name,
            base_url=settings.ollama_base_url,
            temperature=self.temperature,
            num_predict=self.num_predict,
            timeout=settings.ollama_timeout,
            **kwargs
        )
    
    def create_chain(self, 
                    template: str, 
                    chain_name: str = "default",
                    input_variables: Optional[List[str]] = None,
                    use_memory: bool = False) -> LangChainLLMChain:
        """
        Create a new LLM chain with a prompt template.
        
        Args:
            template: Prompt template string
            chain_name: Name for the chain
            input_variables: Variables in the template
            use_memory: Whether to use conversation memory
            
        Returns:
            LangChain LLM chain
        """
        prompt = PromptTemplate(
            template=template,
            input_variables=input_variables or []
        )
        
        memory = self.memory if use_memory else None
        
        chain = LangChainLLMChain(
            llm=self.llm,
            prompt=prompt,
            memory=memory,
            verbose=True
        )
        
        self.chains[chain_name] = chain
        return chain
    
    def predict(self, 
                input_text: str, 
                chain_name: str = "default",
                **kwargs) -> str:
        """
        Make a prediction using a chain.
        
        Args:
            input_text: Input text
            chain_name: Name of the chain to use
            **kwargs: Additional arguments
            
        Returns:
            Generated text
        """
        if chain_name not in self.chains:
            raise ValueError(f"Chain '{chain_name}' not found")
        
        chain = self.chains[chain_name]
        result = chain.predict(input=input_text, **kwargs)
        return result
    
    def chat(self, 
             message: str, 
             chain_name: str = "chat",
             **kwargs) -> str:
        """
        Send a chat message.
        
        Args:
            message: Chat message
            chain_name: Name of the chat chain
            **kwargs: Additional arguments
            
        Returns:
            AI response
        """
        return self.predict(message, chain_name, **kwargs)
    
    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear()
    
    def get_memory_history(self) -> List[BaseMessage]:
        """Get conversation history from memory."""
        return self.memory.chat_memory.messages
    
    def add_callback(self, callback: BaseCallbackHandler):
        """Add a callback handler to the LLM."""
        self.llm.callbacks = [callback]
