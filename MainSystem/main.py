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
    import csv
    from datetime import datetime
    
    # # Load sample data
    # data_dir = Path(__file__).parent / "data"
    # row_file = data_dir / "sample_row.json"
    # story_file = data_dir / "sample_story.txt"
    # 
    # if not row_file.exists() or not story_file.exists():
    #     print(f"Error: Required data files not found in {data_dir}")
    #     print("Please ensure sample_row.json and sample_story.txt exist")
    #     sys.exit(1)
    # 
    # row = json.load(open(row_file))
    # character = row["char"]
    # backstory = row["content"]
    # 
    # story_text = open(story_file).read()
    # 
    # # Process
    # results = process_backstory_and_story(backstory, character, story_text)
    # 
    # # Display results
    # print("\n" + "="*80)
    # print("EXTRACTED CLAIMS/EVENTS:")
    # print("="*80)
    # for i, c in enumerate(results["claims"], 1):
    #     print(f"\n{i}. Subject: {c.subject}")
    #     print(f"   Relation: {c.relation}")
    #     if c.object:
    #         print(f"   Object: {c.object}")
    #     if c.time:
    #         print(f"   Time: {c.time}")
    #     if c.location:
    #         print(f"   Location: {c.location}")
    # 
    # print("\n" + "="*80)
    # print("GENERATED QUESTIONS:")
    # print("="*80)
    # for i, q in enumerate(results["questions"], 1):
    #     print(f"{i}. {q}")
    # 
    # print("\n" + "="*80)
    # print("RAG RETRIEVAL RESULTS:")
    # print("="*80)
    # for q, chunks in results["rag_results"].items():
    #     print(f"\nQ: {q}")
    #     print(f"Retrieved {len(chunks)} chunks:")
    #     for i, c in enumerate(chunks[:3], 1):  # Show top 3 chunks
    #         print(f"  [{i}] {c[:200]}..." if len(c) > 200 else f"  [{i}] {c}")
    
    # Load books
    books_dir = Path(__file__).parent.parent / "Books"
    books = {}
    for book_file in books_dir.glob("*.txt"):
        book_name = book_file.stem  # Get filename without extension
        with open(book_file, 'r', encoding='utf-8') as f:
            books[book_name] = f.read()
    
    print(f"Loaded {len(books)} books:")
    for book_name in books.keys():
        print(f"  - {book_name}")
    
    # Read train.csv
    train_csv_path = Path(__file__).parent.parent / "train.csv"
    
    if not train_csv_path.exists():
        print(f"Error: train.csv not found at {train_csv_path}")
        sys.exit(1)
    
    print(f"\nProcessing rows from {train_csv_path}...")
    
    # Initialize output data structure
    all_results = []
    
    with open(train_csv_path, 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        
        for row_idx, row in enumerate(csv_reader, 1):
            row_id = row.get("id")
            book_name = row.get("book_name")
            character = row.get("char")
            caption = row.get("caption", "")
            backstory = row.get("content")
            label = row.get("label")
            
            print(f"\n{'='*80}")
            print(f"Row {row_idx} (ID: {row_id})")
            print(f"{'='*80}")
            print(f"Book: {book_name}")
            print(f"Character: {character}")
            print(f"Caption: {caption if caption else '(none)'}")
            print(f"Label: {label}")
            
            # Initialize row result
            row_result = {
                "id": row_id,
                "book_name": book_name,
                "character": character,
                "caption": caption,
                "actual_label": label,
                "claims": [],
                "questions": [],
                "rag_results": {},
                "error": None
            }
            
            # Get the book text
            if book_name in books:
                story_text = books[book_name]
                print(f"Book found, length: {len(story_text)} characters")
                
                # Process backstory and story
                try:
                    results = process_backstory_and_story(backstory, character, story_text)
                    
                    print(f"\nExtracted {len(results['claims'])} claims")
                    print(f"Generated {len(results['questions'])} questions")
                    print(f"RAG retrieval completed with {len(results['rag_results'])} results")
                    
                    # Convert claims to serializable format
                    claims_data = []
                    for claim in results['claims']:
                        claims_data.append({
                            "subject": claim.subject,
                            "relation": claim.relation,
                            "object": claim.object,
                            "time": claim.time if hasattr(claim, 'time') else "",
                            "location": claim.location if hasattr(claim, 'location') else ""
                        })
                    
                    row_result["claims"] = claims_data
                    row_result["questions"] = results['questions']
                    
                    # Convert RAG results (Documents to serializable format)
                    rag_results_data = {}
                    for q, chunks in results['rag_results'].items():
                        rag_results_data[q] = chunks
                    row_result["rag_results"] = rag_results_data
                    
                    # # Display results
                    # print("\n" + "-"*80)
                    # print("EXTRACTED CLAIMS/EVENTS:")
                    # print("-"*80)
                    # for i, c in enumerate(results["claims"], 1):
                    #     print(f"\n{i}. Subject: {c.subject}")
                    #     print(f"   Relation: {c.relation}")
                    #     if c.object:
                    #         print(f"   Object: {c.object}")
                    #     if c.time:
                    #         print(f"   Time: {c.time}")
                    #     if c.location:
                    #         print(f"   Location: {c.location}")
                    # 
                    # print("\n" + "-"*80)
                    # print("GENERATED QUESTIONS:")
                    # print("-"*80)
                    # for i, q in enumerate(results["questions"], 1):
                    #     print(f"{i}. {q}")
                    # 
                    # print("\n" + "-"*80)
                    # print("RAG RETRIEVAL RESULTS:")
                    # print("-"*80)
                    # for q, chunks in results["rag_results"].items():
                    #     print(f"\nQ: {q}")
                    #     print(f"Retrieved {len(chunks)} chunks:")
                    #     for i, c in enumerate(chunks[:3], 1):  # Show top 3 chunks
                    #         print(f"  [{i}] {c[:200]}..." if len(c) > 200 else f"  [{i}] {c}")
                    
                except Exception as e:
                    print(f"Error processing row {row_idx}: {str(e)}")
                    row_result["error"] = str(e)
                    import traceback
                    traceback.print_exc()
            else:
                error_msg = f"Book '{book_name}' not found in {books_dir}"
                print(f"Warning: {error_msg}")
                row_result["error"] = error_msg
            
            # Add to results
            all_results.append(row_result)
    
    # Save all results to JSON file
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"results_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")
    
    # Summary statistics
    successful = sum(1 for r in all_results if r["error"] is None)
    failed = sum(1 for r in all_results if r["error"] is not None)
    total_claims = sum(len(r["claims"]) for r in all_results)
    total_questions = sum(len(r["questions"]) for r in all_results)
    
    print(f"\nSummary:")
    print(f"  Total rows processed: {len(all_results)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total claims extracted: {total_claims}")
    print(f"  Total questions generated: {total_questions}")


if __name__ == "__main__":
    main()

