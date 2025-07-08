import os
import sys
from typing import Dict, List
import openai
from chromadb import QueryResult
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .base import ChromaRAG
from src.utils.paper_chunks import papers_to_chunks
from src.utils import get_logger


class PaperRAG(ChromaRAG):
    """RAG system specifically designed for academic papers"""
    
    def __init__(self, chroma_client, collection_name: str = "paper_collection"):
        super().__init__(chroma_client, collection_name)
        self.paper_chunks: Dict[str, List[str]] = {}
    
    def _augment_user_query(self, user_query: str) -> str:
        """Convert user query to better search query using OpenAI"""
        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    ChatCompletionSystemMessageParam(
                        role="system",
                        content="You are a query enhancement system for academic paper search. Convert user questions into optimized search queries that will find relevant academic content. Focus on key technical terms, concepts, and research areas. Keep the query concise but comprehensive."
                    ),
                    ChatCompletionUserMessageParam(
                        role="user",
                        content=f"Convert this question into an optimized search query for academic papers: {user_query}"
                    )
                ],
                max_tokens=200,
                temperature=0.1
            )
            
            improved_query = response.choices[0].message.content.strip()
            return improved_query
            
        except Exception as e:
            self.logger.error(f"Error enhancing query with OpenAI: {e}")
            # Fallback to original query
            return user_query
    
    def _generate_answer(self, user_query: str, rag_results: QueryResult) -> str:
        """Generate a comprehensive answer using RAG results and OpenAI"""
        try:
            # Extract relevant documents from RAG results
            documents = rag_results.get('documents', [[]])[0]  # Get first query results
            metadatas = rag_results.get('metadatas', [[]])[0]
            
            if not documents:
                return "No relevant documents found to answer your question."
            
            # Prepare context from RAG results
            context_parts = []
            for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
                paper_name = metadata.get('paper', 'Unknown paper')
                chunk_index = metadata.get('chunk_index', i)
                context_parts.append(f"From {paper_name} (chunk {chunk_index}):\n{doc}\n")
            
            context = "\n".join(context_parts)
            
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    ChatCompletionSystemMessageParam(
                        role="system",
                        content="You are a research assistant that answers questions based ONLY on the provided context from academic papers. IMPORTANT RULES:\n1. Use ONLY information from the provided context - do not use any external knowledge\n2. Always cite the specific paper name when referencing information\n3. If the context doesn't contain enough information to answer the question, explicitly state this\n4. Include brief, direct quotes from the papers (1-2 sentences max) - use the exact text as it appears, do not modify or rephrase\n5. Use quotation marks for direct excerpts and cite the paper name\n6. Do not reference 'chunks' - instead quote the actual text from the papers\n7. Be precise and accurate with citations\n8. Do not make assumptions or add information not present in the context"
                    ),
                    ChatCompletionUserMessageParam(
                        role="user",
                        content=f"Question: {user_query}\n\nContext from research papers:\n{context}\n\nAnswer the question using ONLY the information provided in the context above. Include brief, direct quotes from the papers (1-2 sentences) using quotation marks, and always cite the specific paper names. Do not reference 'chunks' - quote the actual text as it appears in the papers."
                    )
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content.strip()
            return answer
            
        except Exception as e:
            self.logger.error(f"Error generating answer with OpenAI: {e}")
            return f"Error generating answer: {e}"
    
    def _load_data(self) -> None:
        """Load paper chunks into the collection"""
        self.logger.info("Loading paper chunks...")
        
        # Get paper chunks
        self.paper_chunks = papers_to_chunks()
        
        # Add documents to collection (only if they don't already exist)
        for paper in self.paper_chunks:
            chunks = self.paper_chunks[paper]
            chunk_ids = [f"{paper}_chunk_{i}" for i in range(len(chunks))]
            
            # Check which chunks already exist
            try:
                existing = self.collection.get(ids=chunk_ids)
                existing_ids = set(existing['ids'])
                new_ids = [id for id in chunk_ids if id not in existing_ids]
                new_chunks = [chunks[i] for i, id in enumerate(chunk_ids) if id in new_ids]
                new_metadatas = [{"paper": paper, "chunk_index": i} for i, id in enumerate(chunk_ids) if id in new_ids]
                
                if new_ids:
                    self.collection.add(
                        documents=new_chunks,
                        ids=new_ids,
                        metadatas=new_metadatas
                    )
                    self.logger.info(f"Added {len(new_ids)} new chunks for {paper}")
                else:
                    self.logger.info(f"All chunks for {paper} already exist in collection")
            except:
                # If get() fails, add all chunks
                self.collection.add(
                    documents=chunks,
                    ids=chunk_ids,
                    metadatas=[{"paper": paper, "chunk_index": i} for i in range(len(chunks))]
                )
                self.logger.info(f"Added all chunks for {paper}")
        
        self.logger.info("Paper data loading complete!")