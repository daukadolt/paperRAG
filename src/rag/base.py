from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from chromadb import ClientAPI, Collection
from chromadb import QueryResult
from src.utils import get_logger


class BaseRAG(ABC):
    """Base abstract class for RAG systems"""
    
    @abstractmethod
    def gen(self, user_query: str) -> str:
        """Generate a response to a user query"""
        pass
    
    @abstractmethod
    def setup(self) -> None:
        """Setup the RAG system (initialize collections, load data, etc.)"""
        pass


class ChromaRAG(BaseRAG, ABC):
    """Base class for ChromaDB-based RAG systems"""
    
    def __init__(self, chroma_client: ClientAPI, collection_name: str = "default_collection"):
        self.chroma_client = chroma_client
        self.collection_name = collection_name
        self.collection: Optional[Collection] = None
        self.logger = get_logger(f"PaperRAG.{self.__class__.__name__}")
    
    @abstractmethod
    def _augment_user_query(self, user_query: str) -> str:
        """Augment/improve the user query for better retrieval"""
        pass
    
    @abstractmethod
    def _generate_answer(self, user_query: str, rag_results: QueryResult) -> str:
        """Generate answer from RAG results"""
        pass
    
    def _get_or_create_collection(self) -> Collection:
        """Get existing collection or create new one"""
        try:
            collection = self.chroma_client.get_collection(name=self.collection_name)
            self.logger.info(f"Using existing collection: {self.collection_name}")
            return collection
        except:
            collection = self.chroma_client.create_collection(name=self.collection_name)
            self.logger.info(f"Created new collection: {self.collection_name}")
            return collection
    
    def _query_collection(self, query: str, n_results: int = 10) -> QueryResult:
        """Query the collection with the given query"""
        if not self.collection:
            raise ValueError("Collection not initialized. Call setup() first.")
        
        return self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
    
    def gen(self, user_query: str) -> str:
        """Generate response using the RAG pipeline. Prints improved query."""
        if not self.collection:
            raise ValueError("Collection not initialized. Call setup() first.")
        
        # Step 1: Augment user query
        augmented_query = self._augment_user_query(user_query)
        self.logger.debug(f"Augmented query: {augmented_query}")
        print(f"Improved query: {augmented_query}")
        
        # Step 2: Query collection
        rag_results = self._query_collection(augmented_query)
        
        # Step 3: Generate answer
        answer = self._generate_answer(user_query, rag_results)
        
        return answer
    
    def setup(self) -> None:
        """Setup the RAG system"""
        self.collection = self._get_or_create_collection()
        self._load_data()
    
    @abstractmethod
    def _load_data(self) -> None:
        """Load data into the collection"""
        pass