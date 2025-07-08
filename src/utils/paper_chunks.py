import glob
import pprint

import PyPDF2
import os
import json
import hashlib
from typing import Dict, List
from collections import defaultdict

import chromadb
import openai
from chromadb import QueryResult

from dotenv import load_dotenv
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

PATH_TO_PAPERS = "../../assets/papers"
CACHE_FILE = "../../assets/paper_chunks_cache.json"

def papers_to_chunks(chunk_size=800, chunk_overlap=200) -> Dict[str, List[str]]:
    # Check if cache exists and is valid
    cache_data = load_cache(chunk_size, chunk_overlap)
    if cache_data:
        print("Using cached chunks - no recomputation needed!")
        return cache_data
    
    print("Cache not found or invalid - processing PDFs...")
    out = defaultdict(list)
    for file in glob.glob(f'{PATH_TO_PAPERS}/*.pdf'):
        try:
            print(f"Reading: {os.path.basename(file)}")
            
            # Open PDF file in binary mode
            with open(file, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                
                # Extract text from all pages
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                
                print(f"Extracted {len(text)} characters from {len(pdf_reader.pages)} pages")
                
                # Create chunks with overlap
                chunks = []
                i = 0
                while i < len(text):
                    chunk_start = i
                    chunk_end = min(i + chunk_size, len(text))
                    
                    # Extract the chunk
                    chunk = text[chunk_start:chunk_end]
                    chunks.append(chunk)
                    
                    # Move to next chunk with overlap
                    i += (chunk_size - chunk_overlap)
                
                # Store chunks in output dictionary
                out[os.path.basename(file)] = chunks

        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    result = dict(out)
    
    # Save to cache
    save_cache(result, chunk_size, chunk_overlap)
    
    return result


def get_file_hash(filepath: str) -> str:
    """Get MD5 hash of file modification time and size"""
    stat = os.stat(filepath)
    # Combine modification time and file size for hash
    content = f"{stat.st_mtime}_{stat.st_size}"
    return hashlib.md5(content.encode()).hexdigest()


def load_cache(chunk_size: int, chunk_overlap: int) -> Dict[str, List[str]] | None:
    """Load cached chunks if valid"""
    if not os.path.exists(CACHE_FILE):
        return None
    
    try:
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)
        
        # Check if cache parameters match
        if cache.get('chunk_size') != chunk_size or cache.get('chunk_overlap') != chunk_overlap:
            return None
        
        # Check if all files in cache still exist and haven't changed
        file_hashes = cache.get('file_hashes', {})
        for file in glob.glob(f'{PATH_TO_PAPERS}/*.pdf'):
            filename = os.path.basename(file)
            current_hash = get_file_hash(file)
            
            if filename not in file_hashes or file_hashes[filename] != current_hash:
                return None
        
        return cache.get('chunks', {})
        
    except Exception as e:
        print(f"Error loading cache: {e}")
        return None


def save_cache(chunks: Dict[str, List[str]], chunk_size: int, chunk_overlap: int):
    """Save chunks to cache with file hashes"""
    # Calculate file hashes
    file_hashes = {}
    for file in glob.glob(f'{PATH_TO_PAPERS}/*.pdf'):
        filename = os.path.basename(file)
        file_hashes[filename] = get_file_hash(file)
    
    cache_data = {
        'chunks': chunks,
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'file_hashes': file_hashes
    }
    
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache_data, f, indent=2)
        print(f"Cache saved to {CACHE_FILE}")
    except Exception as e:
        print(f"Error saving cache: {e}")


def convert_query_with_openai(user_query: str) -> str:
    """Convert user query to a better search query using OpenAI"""
    try:
        # You'll need to set your OpenAI API key
        # openai.api_key = "your-api-key-here"
        
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                ChatCompletionSystemMessageParam(
                    role="system",
                    content="You are a helpful assistant that converts user queries into optimized search queries for academic papers. Focus on extracting key technical terms, concepts, and methodologies that would appear in research papers. Make the query more specific and technical while preserving the original intent."
                ),
                ChatCompletionUserMessageParam(
                    role="user",
                    content=f"Convert this user query into a better search query for academic papers: '{user_query}'"
                )
            ],
            max_tokens=100,
            temperature=0.3
        )
        
        improved_query = response.choices[0].message.content.strip()
        print(f"Original query: {user_query}")
        print(f"Improved query: {improved_query}")
        return improved_query
        
    except Exception as e:
        print(f"Error converting query with OpenAI: {e}")
        print("Using original query instead")
        return user_query


def generate_answer_with_rag(user_query: str, rag_results: QueryResult) -> str:
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
        print(f"Error generating answer with OpenAI: {e}")
        return f"Error generating answer: {e}"


if __name__ == "__main__":
    load_dotenv()
    paper_chunks = papers_to_chunks()
    chroma_client = chromadb.PersistentClient()
    
    # Check if collection exists, create if it doesn't
    try:
        collection = chroma_client.get_collection(name="paper_collection")
        print("Using existing collection: paper_collection")
        collection_exists = True
    except:
        collection = chroma_client.create_collection(name="paper_collection")
        print("Created new collection: paper_collection")
        collection_exists = False
    
    # Add documents to collection (only if they don't already exist)
    for paper in paper_chunks:
        chunks = paper_chunks[paper]
        chunk_ids = [f"{paper}_chunk_{i}" for i in range(len(chunks))]
        
        if collection_exists:
            # Check which chunks already exist
            try:
                existing = collection.get(ids=chunk_ids)
                existing_ids = set(existing['ids'])
                new_ids = [id for id in chunk_ids if id not in existing_ids]
                new_chunks = [chunks[i] for i, id in enumerate(chunk_ids) if id in new_ids]
                new_metadatas = [{"paper": paper, "chunk_index": i} for i, id in enumerate(chunk_ids) if id in new_ids]
                
                if new_ids:
                    collection.add(
                        documents=new_chunks,
                        ids=new_ids,
                        metadatas=new_metadatas
                    )
                    print(f"Added {len(new_ids)} new chunks for {paper}")
                else:
                    print(f"All chunks for {paper} already exist in collection")
            except:
                # If get() fails, add all chunks
                collection.add(
                    documents=chunks,
                    ids=chunk_ids,
                    metadatas=[{"paper": paper, "chunk_index": i} for i in range(len(chunks))]
                )
                print(f"Added all chunks for {paper}")
        else:
            # New collection, add all chunks
            collection.add(
                documents=chunks,
                ids=chunk_ids,
                metadatas=[{"paper": paper, "chunk_index": i} for i in range(len(chunks))]
            )
            print(f"Added all chunks for {paper}")
    
    user_query = "Best way to design an agent"
    
    # Convert user query to better search query using OpenAI
    improved_query = convert_query_with_openai(user_query)

    # Get RAG results from ChromaDB
    results = collection.query(
        query_texts=[improved_query],
        n_results=10,
        include=["documents", "metadatas", "distances"],
    )
    
    print("\n" + "="*50)
    print("RAG RESULTS:")
    print("="*50)
    pprint.pprint(results)
    
    # Generate comprehensive answer using RAG results
    print("\n" + "="*50)
    print("GENERATED ANSWER:")
    print("="*50)
    answer = generate_answer_with_rag(user_query, results)
    print(answer)