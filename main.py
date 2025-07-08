#!/usr/bin/env python3
"""
PaperRAG - Interactive RAG System for Academic Papers
Allows users to ask questions and get answers based on the paper collection.
"""

import sys
import os
from typing import Optional

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import chromadb
from dotenv import load_dotenv
from src.rag.PaperRAG import PaperRAG


class PaperRAGApp:
    """Main application class for PaperRAG"""
    
    def __init__(self):
        self.rag_system: Optional[PaperRAG] = None
    
    def setup(self) -> bool:
        """Setup the RAG system"""
        try:
            print("Setting up PaperRAG system...")
            
            # Initialize ChromaDB client
            chroma_client = chromadb.PersistentClient(path="chroma")
            
            # Create and setup PaperRAG
            self.rag_system = PaperRAG(chroma_client)
            self.rag_system.setup()
            
            print("✅ System ready! You can now ask questions.")
            return True
            
        except Exception as e:
            print(f"❌ Error setting up system: {e}")
            return False
    
    def process_query(self, user_query: str) -> str:
        """Process a user query through the RAG pipeline"""
        if not self.rag_system:
            return "System not initialized. Please run setup() first."
        
        print(f"\n🔍 Processing query: '{user_query}'")
        print("-" * 50)
        
        try:
            answer = self.rag_system.gen(user_query)
            return answer
        except Exception as e:
            return f"❌ Error processing query: {e}"
    
    def run_interactive(self):
        """Run the interactive question-answering loop"""
        print("=" * 60)
        print("📚 PaperRAG - Interactive Research Assistant")
        print("=" * 60)
        print("This system answers questions based on your academic paper collection.")
        print("Type 'quit' or 'exit' to end the session.")
        print("-" * 60)
        
        # Setup the system
        if not self.setup():
            return
        
        # Interactive loop
        while True:
            try:
                # Get user input
                user_query = input("\n🤔 Your question: ").strip()
                
                # Handle special commands
                if user_query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif not user_query:
                    print("Please enter a question.")
                    continue
                
                # Process the query
                answer = self.process_query(user_query)
                
                # Display the answer
                print("\n" + "=" * 60)
                print("📝 ANSWER:")
                print("=" * 60)
                print(answer)
                print("=" * 60)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again.")


def main():
    """Main entry point"""
    # Load environment variables
    load_dotenv()
    
    # Create and run the application
    app = PaperRAGApp()
    app.run_interactive()


if __name__ == "__main__":
    main() 