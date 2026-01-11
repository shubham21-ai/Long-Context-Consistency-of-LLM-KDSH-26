"""
Main pipeline for narrative consistency checking.

This script processes character backstories, extracts events, generates questions,
and queries the story using RAG to verify consistency.
"""

import json
import sys
import tempfile
import os
from pathlib import Path
from backstory.claim_extractor import extract_claims
from questions.question_generator import generate_questions_from_event
from rag import RAG


def process_backstory_and_story(backstory_text: str, character: str, story_text: str, 
                                model_llm: str = "gemini-2.0-flash", 
                                model_embedding: str = "models/embedding-001",
                                k: int = 10):
    """
    Process a character backstory and story to generate questions and retrieve relevant chunks.
    
    Args:
        backstory_text: The character backstory text
        character: The main character name
        story_text: The full story text to search
        model_llm: Gemini LLM model name
        model_embedding: Gemini embedding model name
        k: Number of chunks to retrieve per question
        
    Returns:
        Dictionary with claims, questions, and RAG results
    """
    # Create temporary file for story text (RAG class loads from directory)
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, "story.txt")
    
    try:
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(story_text)
        
        print(f"Story text length: {len(story_text)} characters")
        
        # Initialize RAG system
        print("Initializing RAG system...")
        rag = RAG(
            model_llm=model_llm,
            model_embedding=model_embedding,
            data_dir=temp_dir
        )
        
        # Load models
        print("Loading Gemini models...")
        rag.load_model()
        print("Models loaded successfully")
        
        # Load data from directory
        print("Loading story data...")
        rag.load_data()
        print(f"Loaded {len(rag.documents)} document(s)")
        
        # Split data into chunks
        print("Splitting data into chunks...")
        rag.split_data()
        print(f"Created {len(rag.texts)} chunks")
        
        # Create embeddings
        print("Creating embeddings...")
        rag.embed_data()
        print(f"Created embeddings with shape: {rag.embeddings.shape}")
        
        # Create search indices (FAISS + BM25)
        print("Building search indices (FAISS + BM25)...")
        rag.create_index()
        print("Indices built successfully")
        
        # Extract events/claims from backstory
        print(f"\nExtracting claims from backstory for character: {character}")
        claims = extract_claims(backstory_text, character)
        print(f"Extracted {len(claims)} claims/events")
        
        # Generate questions from events
        print("\nGenerating questions from events...")
        questions = []
        for i, claim in enumerate(claims, 1):
            print(f"  Processing claim {i}/{len(claims)}: {claim.subject} {claim.relation}")
            event_questions = generate_questions_from_event(claim, character)
            questions.extend(event_questions)
            print(f"    Generated {len(event_questions)} questions")
        
        print(f"\nTotal questions generated: {len(questions)}")
        
        # Query RAG system for each question
        print(f"\nQuerying RAG system (retrieving top {k} chunks per question)...")
        rag_results = {}
        for i, question in enumerate(questions, 1):
            print(f"  Querying {i}/{len(questions)}: {question[:60]}...")
            retrieved_docs = rag.retrieve_data(question)
            # Extract text content from LangChain Document objects
            # retrieve_data returns list of Document objects from self.texts
            retrieved_chunks = []
            for doc in retrieved_docs[:k]:
                if hasattr(doc, 'page_content'):
                    retrieved_chunks.append(doc.page_content)
                elif isinstance(doc, str):
                    retrieved_chunks.append(doc)
                else:
                    retrieved_chunks.append(str(doc))
            rag_results[question] = retrieved_chunks
        
        print("RAG queries completed")
        
        return {
            "claims": claims,
            "questions": questions,
            "rag_results": rag_results
        }
        
    finally:
        # Clean up temporary directory
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Main entry point for the pipeline."""
    # Load sample data
    data_dir = Path(__file__).parent / "data"
    row_file = data_dir / "sample_row.json"
    story_file = data_dir / "sample_story.txt"
    
    if not row_file.exists() or not story_file.exists():
        print(f"Error: Required data files not found in {data_dir}")
        print("Please ensure sample_row.json and sample_story.txt exist")
        sys.exit(1)
    
    row = json.load(open(row_file))
    character = row["char"]
    backstory = row["content"]
    
    story_text = open(story_file).read()
    
    # Process
    results = process_backstory_and_story(backstory, character, story_text)
    
    # Display results
    print("\n" + "="*80)
    print("EXTRACTED CLAIMS/EVENTS:")
    print("="*80)
    for i, c in enumerate(results["claims"], 1):
        print(f"\n{i}. Subject: {c.subject}")
        print(f"   Relation: {c.relation}")
        if c.object:
            print(f"   Object: {c.object}")
        if c.time:
            print(f"   Time: {c.time}")
        if c.location:
            print(f"   Location: {c.location}")
    
    print("\n" + "="*80)
    print("GENERATED QUESTIONS:")
    print("="*80)
    for i, q in enumerate(results["questions"], 1):
        print(f"{i}. {q}")
    
    print("\n" + "="*80)
    print("RAG RETRIEVAL RESULTS:")
    print("="*80)
    for q, chunks in results["rag_results"].items():
        print(f"\nQ: {q}")
        print(f"Retrieved {len(chunks)} chunks:")
        for i, c in enumerate(chunks[:3], 1):  # Show top 3 chunks
            print(f"  [{i}] {c[:200]}..." if len(c) > 200 else f"  [{i}] {c}")


if __name__ == "__main__":
    main()

