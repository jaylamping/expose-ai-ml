"""
Pre-built prompt templates for common use cases.
"""
from typing import Dict, Any
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate


class PromptTemplates:
    """Collection of useful prompt templates."""
    
    # Text Analysis Templates
    SUMMARIZE = PromptTemplate(
        input_variables=["text"],
        template="Please provide a concise summary of the following text:\n\n{text}\n\nSummary:"
    )
    
    EXTRACT_ENTITIES = PromptTemplate(
        input_variables=["text"],
        template="Extract all named entities (people, places, organizations, etc.) from the following text:\n\n{text}\n\nEntities:"
    )
    
    SENTIMENT_ANALYSIS = PromptTemplate(
        input_variables=["text"],
        template="Analyze the sentiment of the following text and provide a score from -1 (very negative) to 1 (very positive):\n\n{text}\n\nSentiment:"
    )
    
    # Code Generation Templates
    CODE_REVIEW = PromptTemplate(
        input_variables=["code"],
        template="Review the following code and provide feedback on:\n1. Code quality and best practices\n2. Potential bugs or issues\n3. Performance improvements\n4. Security concerns\n\nCode:\n{code}\n\nReview:"
    )
    
    CODE_EXPLANATION = PromptTemplate(
        input_variables=["code"],
        template="Explain what the following code does in simple terms:\n\n{code}\n\nExplanation:"
    )
    
    # Creative Writing Templates
    STORY_GENERATOR = PromptTemplate(
        input_variables=["prompt", "genre", "length"],
        template="Write a {length} {genre} story based on this prompt: {prompt}\n\nStory:"
    )
    
    POEM_GENERATOR = PromptTemplate(
        input_variables=["topic", "style"],
        template="Write a {style} poem about {topic}:\n\nPoem:"
    )
    
    # Data Analysis Templates
    DATA_INSIGHTS = PromptTemplate(
        input_variables=["data_description", "question"],
        template="Given this data: {data_description}\n\nAnswer this question: {question}\n\nAnalysis:"
    )
    
    # Chat Templates
    HELPFUL_ASSISTANT = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful, knowledgeable, and friendly AI assistant. Provide accurate and useful responses to user questions."),
        ("human", "{input}")
    ])
    
    TECHNICAL_EXPERT = ChatPromptTemplate.from_messages([
        ("system", "You are a technical expert with deep knowledge in software development, machine learning, and data science. Provide detailed, accurate technical explanations."),
        ("human", "{input}")
    ])
    
    CREATIVE_WRITER = ChatPromptTemplate.from_messages([
        ("system", "You are a creative writer with expertise in storytelling, poetry, and creative expression. Help users with their creative writing needs."),
        ("human", "{input}")
    ])
    
    @classmethod
    def get_template(cls, name: str) -> PromptTemplate:
        """Get a template by name."""
        templates = {
            "summarize": cls.SUMMARIZE,
            "extract_entities": cls.EXTRACT_ENTITIES,
            "sentiment_analysis": cls.SENTIMENT_ANALYSIS,
            "code_review": cls.CODE_REVIEW,
            "code_explanation": cls.CODE_EXPLANATION,
            "story_generator": cls.STORY_GENERATOR,
            "poem_generator": cls.POEM_GENERATOR,
            "data_insights": cls.DATA_INSIGHTS,
        }
        
        if name not in templates:
            raise ValueError(f"Template '{name}' not found. Available templates: {list(templates.keys())}")
        
        return templates[name]
    
    @classmethod
    def get_chat_template(cls, name: str) -> ChatPromptTemplate:
        """Get a chat template by name."""
        chat_templates = {
            "helpful_assistant": cls.HELPFUL_ASSISTANT,
            "technical_expert": cls.TECHNICAL_EXPERT,
            "creative_writer": cls.CREATIVE_WRITER,
        }
        
        if name not in chat_templates:
            raise ValueError(f"Chat template '{name}' not found. Available templates: {list(chat_templates.keys())}")
        
        return chat_templates[name]
