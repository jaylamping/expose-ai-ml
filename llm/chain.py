"""
LangChain integration with Ollama for local LLM applications.
"""
from typing import Dict, Optional, List, Union, Protocol, runtime_checkable
from langchain_ollama import OllamaLLM
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.callbacks.base import BaseCallbackHandler

from config.settings import settings


@runtime_checkable
class LLMProtocol(Protocol):
    """Protocol for LLM-like objects."""
    def __call__(self, text: str) -> str:
        """Call the LLM with text input."""
        ...


@runtime_checkable
class MemoryProtocol(Protocol):
    """Protocol for memory-like objects."""
    def clear(self) -> None:
        """Clear the memory."""
        ...
    
    @property
    def chat_memory(self) -> 'ChatMemoryProtocol':
        """Get chat memory."""
        ...


@runtime_checkable
class ChatMemoryProtocol(Protocol):
    """Protocol for chat memory objects."""
    @property
    def messages(self) -> List[BaseMessage]:
        """Get messages."""
        ...


@runtime_checkable
class ChainProtocol(Protocol):
    """Protocol for chain-like objects."""
    def __call__(self, input: str) -> str:
        """Call the chain with input."""
        ...


class SimpleMemory:
    """Simple memory implementation."""
    def __init__(self):
        self._messages: List[BaseMessage] = []
    
    def clear(self) -> None:
        """Clear the memory."""
        self._messages.clear()
    
    @property
    def chat_memory(self) -> 'SimpleChatMemory':
        """Get chat memory."""
        return SimpleChatMemory(self._messages)


class SimpleChatMemory:
    """Simple chat memory implementation."""
    def __init__(self, messages: List[BaseMessage]):
        self._messages = messages
    
    @property
    def messages(self) -> List[BaseMessage]:
        """Get messages."""
        return self._messages


class SimpleChain:
    """Simple chain implementation that follows ChainProtocol."""
    def __init__(self, llm: LLMProtocol, prompt: PromptTemplate, memory: Optional[MemoryProtocol] = None):
        self.llm = llm
        self.prompt = prompt
        self.memory = memory
    
    def __call__(self, input: str) -> str:
        """Call the chain with input."""
        # Format the prompt with input variables
        formatted_prompt = self.prompt.format(input=input)
        # Use the LLM directly
        result = self.llm(formatted_prompt)
        return result


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
        self.memory = SimpleMemory()
        self.chains: Dict[str, ChainProtocol] = {}
    
    def _create_llm(self, **kwargs) -> LLMProtocol:
        """Create the Ollama LLM instance."""
        return OllamaLLM(
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
                    use_memory: bool = False) -> ChainProtocol:
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
        
        # Create a chain that follows the ChainProtocol
        chain = SimpleChain(self.llm, prompt, memory)
        
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
        result = chain(input_text)
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
