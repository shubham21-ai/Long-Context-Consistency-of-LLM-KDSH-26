"""
Test Framework for Narrative Consistency Checking

This framework:
1. Ingests novels from the Books folder
2. Processes test.csv with character backstories
3. Generates questions and queries the story
4. Evaluates accuracy by checking if retrieved chunks contain relevant information
"""

import csv
import json
import sys
import tempfile
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Suppress tokenizers warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Ensure output is not buffered
try:
    sys.stdout.reconfigure(line_buffering=True)
except:
    pass

# Add lohiya_code to path
sys.path.insert(0, str(Path(__file__).parent))

from backstory.claim_extractor import extract_claims
from questions.question_generator import generate_questions_from_backstory
from rag import RAG
from evaluator import evaluate_consistency, apply_decision_rule


class NarrativeConsistencyTester:
    """Test framework for narrative consistency checking."""
    
    def __init__(self, books_dir: str = None, test_csv: str = None, 
                 model_llm: str = "gemini-3.0-flash-preview", 
                 model_embedding: str = "models/embedding-001"):
        """
        Initialize the test framework.
        
        Args:
            books_dir: Directory containing novel text files (default: ../Books)
            test_csv: Path to test CSV file with backstories (default: ../test.csv)
            model_llm: Gemini LLM model name
            model_embedding: Gemini embedding model name
        """
        # Set default paths relative to lohiya_code directory
        if books_dir is None:
            books_dir = Path(__file__).parent.parent / "Books"
        if test_csv is None:
            test_csv = Path(__file__).parent.parent / "test.csv"
        
        self.books_dir = Path(books_dir)
        self.test_csv = Path(test_csv)
        self.books_cache = {}  # Cache loaded books
        self.rag_cache = {}  # Cache prepared RAG instances per book
        self.model_llm = model_llm
        self.model_embedding = model_embedding
        
    def load_book(self, book_name: str) -> str:
        """
        Load a book from the Books directory.
        
        Args:
            book_name: Name of the book (filename without .txt)
            
        Returns:
            Book text content
        """
        if book_name in self.books_cache:
            return self.books_cache[book_name]
        
        # Try to find the book file (case-insensitive)
        book_files = list(self.books_dir.glob("*.txt"))
        book_file = None
        
        for f in book_files:
            if book_name.lower() in f.stem.lower() or f.stem.lower() in book_name.lower():
                book_file = f
                break
        
        if not book_file:
            raise FileNotFoundError(f"Book '{book_name}' not found in {self.books_dir}")
        
        print(f"Loading book: {book_file.name}", flush=True)
        with open(book_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        self.books_cache[book_name] = content
        return content
    
    def load_test_data(self) -> List[Dict]:
        """
        Load test data from CSV file.
        
        Returns:
            List of dictionaries with test cases
        """
        test_cases = []
        with open(self.test_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                test_cases.append({
                    'id': row.get('id', ''),
                    'book_name': row.get('book_name', ''),
                    'char': row.get('char', ''),
                    'content': row.get('content', ''),
                    'label': row.get('label', '0')  # Expected label (1 = consistent, 0 = contradict)
                })
        return test_cases
    
    def process_test_case(self, test_case: Dict, k: int = 15, output_file=None) -> Dict:
        """
        Single-pass processing WITHOUT claim extraction:
        Backstory → Questions → RAG → Evaluation → Decision
        
        Args:
            test_case: Test case dictionary
            k: Number of chunks to retrieve
            output_file: File object to write formatted output to
        """
        book_name = test_case['book_name']
        character = test_case['char']
        backstory = test_case['content']
        expected_label = test_case.get('label', '0')

        rag = None
        temp_dir = None
        use_cached = False

        try:
            # ------------------------------------------------------------
            # Load book
            # ------------------------------------------------------------
            story_text = self.load_book(book_name)

            # ------------------------------------------------------------
            # Prepare or reuse RAG
            # ------------------------------------------------------------
            if book_name in self.rag_cache:
                rag = self.rag_cache[book_name]['rag']
                use_cached = True
                print(f"Reusing cached RAG for book: {book_name}", flush=True)
            else:
                temp_dir = tempfile.mkdtemp()
                with open(os.path.join(temp_dir, "story.txt"), "w", encoding="utf-8") as f:
                    f.write(story_text)

                rag = RAG(
                    model_llm=self.model_llm,
                    model_embedding=self.model_embedding,
                    data_dir=temp_dir
                )
                rag.load_model()

                try:
                    rag.load_data()
                except Exception:
                    from langchain_core.documents import Document
                    rag.documents = [Document(page_content=story_text)]

                rag.split_data()
                rag.embed_data()
                rag.create_index()

                self.rag_cache[book_name] = {'rag': rag, 'temp_dir': temp_dir}

            # ------------------------------------------------------------
            # Generate questions DIRECTLY from backstory
            # ------------------------------------------------------------
            print("Generating questions directly from backstory...", flush=True)

            questions = generate_questions_from_backstory(
                backstory=backstory,
                main_character=character
            )

            if not questions:
                raise ValueError("No questions generated from backstory")

            # ------------------------------------------------------------
            # Retrieve + Evaluate inline
            # ------------------------------------------------------------
            evaluations = []
            rag_results = {}

            for i, q in enumerate(questions, 1):
                print(f"  [{i}/{len(questions)}] {q[:60]}...", flush=True)
                
                if output_file:
                    output_file.write(f"\n  Question [{i}/{len(questions)}]:\n")
                    output_file.write(f"  {q}\n")

                try:
                    docs = rag.retrieve_data(q, k=k)
                    chunks = [
                        d.page_content if hasattr(d, "page_content") else str(d)
                        for d in docs
                    ]

                    rag_results[q] = chunks
                    
                    if output_file:
                        output_file.write(f"\n  RAG Retrieved {len(chunks)} chunks (showing top 1):\n")
                        output_file.write(f"  {'-'*70}\n")
                        if chunks:
                            output_file.write(f"\n  TOP CHUNK:\n")
                            output_file.write(f"  {'-'*70}\n")
                            output_file.write(f"{chunks[0]}\n")
                            output_file.write(f"  {'-'*70}\n")

                    eval_result = evaluate_consistency(q, chunks, backstory, character=character)
                    evaluations.append(eval_result)
                    
                    if output_file:
                        output_file.write(f"\n  Evaluation Result:\n")
                        output_file.write(f"    Verdict: {eval_result.get('verdict', 'uncertain')}\n")
                        output_file.write(f"    Answer: {eval_result.get('answer', 'N/A')}\n")
                        output_file.write(f"    Confidence: {eval_result.get('confidence', 0.0)}\n")
                        output_file.write(f"    Reasoning: {eval_result.get('reasoning', 'N/A')}\n")

                    if eval_result.get("verdict") == "contradict":
                        print("    ⚠️ contradict detected", flush=True)
                        print(f"       Reason: {eval_result.get('reasoning', 'N/A')}", flush=True)

                    print(f"    → {eval_result.get('verdict', 'uncertain')}", flush=True)
                except Exception as q_error:
                    print(f"    ✗ Error evaluating question: {str(q_error)[:100]}", flush=True)
                    import traceback
                    traceback.print_exc()
                    
                    if output_file:
                        output_file.write(f"\n  ERROR during evaluation:\n")
                        output_file.write(f"    {str(q_error)}\n")
                    
                    evaluations.append({
                        'verdict': 'uncertain',
                        'answer': 'ERROR',
                        'reasoning': str(q_error)
                    })

            # ------------------------------------------------------------
            # Decision rule: If ANY evaluation is 'contradict', entire row is 'contradict'
            # ------------------------------------------------------------
            has_any_contradict = any(e.get('verdict') == 'contradict' for e in evaluations)
            verdict = 'contradict' if has_any_contradict else 'consistent'
            
            predicted_label = '0' if verdict == 'contradict' else '1'
            label_match = (predicted_label == expected_label)


            contradict_reasons = [
            {
                "question": q,
                "reasoning": e.get("reasoning", ""),
                "answer": e.get("answer", "")
            }
            for q, e in zip(questions, evaluations)
            if e.get("verdict") == "contradict"
        ]



            # ------------------------------------------------------------
            # Print final result
            # ------------------------------------------------------------
            print("\n================ ROW RESULT ================", flush=True)
            print(f"Book        : {book_name}", flush=True)
            print(f"Character   : {character}", flush=True)
            print(f"Verdict     : {verdict}", flush=True)
            print(f"Expected    : {expected_label}", flush=True)
            print(f"Predicted   : {predicted_label}", flush=True)
            print(f"Match       : {'✓' if label_match else '✗'}", flush=True)
            print("===========================================\n", flush=True)
            
            if output_file:
                output_file.write("\n" + "="*60 + "\n")
                output_file.write(f"FINAL ROW RESULT\n")
                output_file.write("="*60 + "\n")
                output_file.write(f"Book: {book_name}\n")
                output_file.write(f"Character: {character}\n")
                output_file.write(f"Verdict: {verdict}\n")
                output_file.write(f"Expected Label: {expected_label}\n")
                output_file.write(f"Predicted Label: {predicted_label}\n")
                output_file.write(f"Match: {'YES' if label_match else 'NO'}\n")
                output_file.write("="*60 + "\n\n")
                output_file.flush()
            print(f"Predicted   : {predicted_label}", flush=True)
            print(f"Match       : {'✓' if label_match else '✗'}", flush=True)
            print("===========================================\n", flush=True)


            return {
                'test_case': test_case,
                'questions': questions,
                'rag_results': rag_results,
                'evaluations': evaluations,
                'decision_rule': decision,
                'predicted_label': predicted_label,
                'expected_label': expected_label,
                'label_match': label_match,
                "contradict_reasons": contradict_reasons,
                'success': True
            }

        except Exception as e:
            import traceback
            return {
                'test_case': test_case,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'success': False
            }

        finally:
            if temp_dir and not use_cached:
                shutil.rmtree(temp_dir, ignore_errors=True)



    
    def run_tests(self, k: int = 8, max_tests: int = None) -> Dict:
        """
        Run all tests in a single pass:
        each row is fully processed (retrieval + evaluation + decision)
        before moving to the next.
        """
        print("=" * 80)
        print("NARRATIVE CONSISTENCY TEST FRAMEWORK (SINGLE PASS)")
        print("=" * 80)

        # ------------------------------------------------------------
        # Load test data
        # ------------------------------------------------------------
        print(f"\nLoading test data from {self.test_csv}...", flush=True)
        test_cases = self.load_test_data()
        if max_tests:
            test_cases = test_cases[:max_tests]
        print(f"Loaded {len(test_cases)} test cases", flush=True)

        results = []
        correct_predictions = 0
        total_predictions = 0

        # ------------------------------------------------------------
        # Process each row ONCE
        # ------------------------------------------------------------
        for idx, test_case in enumerate(test_cases, 1):
            print(f"\n{'=' * 80}", flush=True)
            print(f"Row {idx}/{len(test_cases)}", flush=True)
            print(f"Book      : {test_case['book_name']}", flush=True)
            print(f"Character : {test_case['char']}", flush=True)
            print(f"Backstory : {test_case['content'][:100]}...", flush=True)
            print(f"{'=' * 80}", flush=True)

            result = self.process_test_case(test_case, k=k, output_file=output_f)
            results.append(result)

            if not result.get("success", False):
                print("✗ Row failed — skipping accuracy update", flush=True)
                continue

            total_predictions += 1

            if result.get("label_match", False):
                correct_predictions += 1

        # ------------------------------------------------------------
        # Final metrics
        # ------------------------------------------------------------
        accuracy = (correct_predictions / total_predictions * 100) if total_predictions else 0.0

        print(f"\n{'=' * 80}", flush=True)
        print("COMPLETED ALL TEST CASES", flush=True)
        print(f"{'=' * 80}", flush=True)
        print(f"Total rows processed : {len(test_cases)}", flush=True)
        print(f"Valid predictions    : {total_predictions}", flush=True)
        print(f"Correct predictions  : {correct_predictions}", flush=True)
        print(f"Accuracy             : {accuracy:.2f}%", flush=True)

        return {
            "total_tests": len(test_cases),
            "total_predictions": total_predictions,
            "correct_predictions": correct_predictions,
            "accuracy": accuracy,
            "results": results,
        }

    
    def print_summary(self, summary: Dict):
        """Print test summary and results."""
        print("\n" + "="*80, flush=True)
        print("TEST SUMMARY", flush=True)
        print("="*80, flush=True)
        print(f"Total Test Cases: {summary['total_tests']}", flush=True)
        print(f"Successful Tests: {summary['successful_tests']}", flush=True)
        print(f"Consistent Evaluations: {summary.get('total_consistent', 0)}", flush=True)
        print(f"Contradict Evaluations: {summary.get('total_contradict', 0)}", flush=True)
        print(f"Uncertain Evaluations: {summary.get('total_uncertain', 0)}", flush=True)
        print(f"\n{'='*80}", flush=True)
        print(f"ACCURACY: {summary['accuracy']:.2f}%", flush=True)
        print(f"Correct Predictions: {summary.get('correct_predictions', 0)}/{summary.get('total_predictions', 0)}", flush=True)
        print("="*80, flush=True)
        
        # Detailed results
        print("\nDETAILED RESULTS:", flush=True)
        for i, result in enumerate(summary['results'], 1):
            if not result.get('success', False):
                error_msg = result.get('error', 'Unknown error')
                # Truncate long error messages
                if len(error_msg) > 200:
                    error_msg = error_msg[:200] + "..."
                print(f"\nTest {i}: FAILED - {error_msg}", flush=True)
                continue
            
            test_case = result['test_case']
            expected_label = result.get('expected_label', '0')
            expected_text = "contradict" if expected_label == '0' else "consistent"
            
            print(f"\nTest {i}: {test_case['book_name']} - {test_case['char']}", flush=True)
            print(f"  Expected: {expected_text} (label: {expected_label})", flush=True)
            
            if 'decision_rule' in result:
                dr = result['decision_rule']
                verdict = dr['verdict']
                confidence = dr['confidence']
                is_correct = (expected_label == '0' and verdict == 'contradict') or \
                            (expected_label == '1' and verdict == 'consistent')
                status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
                
                print(f"  Predicted: {verdict} (confidence: {confidence:.2f}) {status}", flush=True)
                print(f"  Reason: {dr['verdict_reason']}", flush=True)
                
                signals = dr.get('signals', {})
                print(f"  Evaluations: {signals.get('consistent_count', 0)} consistent, "
                      f"{signals.get('contradict_count', 0)} contradict, "
                      f"{signals.get('uncertain_count', 0)} uncertain", flush=True)
            
            if 'evaluations' in result:
                contradict_evals = [e for e in result['evaluations'] if e.get('verdict') == 'contradict']
                if contradict_evals:
                    print(f"  ⚠️  contradict evaluations: {len(contradict_evals)}", flush=True)
                    for eval_result in contradict_evals[:3]:  # Show first 3
                        print(f"    - {eval_result.get('answer', 'N/A')[:60]}...", flush=True)


def main():
    """Main entry point for test framework.

    NOTE: The original CLI-driven `main()` is preserved below as a module-level
    string (commented-out) to avoid deleting any code, per request. A new main
    implementation follows which reads `train.csv`, processes rows using the
    existing `NarrativeConsistencyTester`, and saves results to JSON.
    """

    # --- ORIGINAL MAIN (commented-out) ---
    ORIGINAL_MAIN = '''
    def main():
        """Main entry point for test framework."""
        import argparse
        
        # Ensure unbuffered output
        import sys
        sys.stdout = sys.__stdout__  # Reset to ensure clean output
        os.environ['PYTHONUNBUFFERED'] = '1'
        
        parser = argparse.ArgumentParser(description='Test narrative consistency framework')
        parser.add_argument('--books-dir', default=None, help='Directory containing books (default: ../Books)')
        parser.add_argument('--test-csv', default=None, help='Path to test CSV file (default: ../test.csv)')
        parser.add_argument('--k', type=int, default=10, help='Number of chunks to retrieve')
        parser.add_argument('--max-tests', type=int, default=None, help='Maximum number of tests to run')
        
        args = parser.parse_args()
        
        # Create tester
        tester = NarrativeConsistencyTester(
            books_dir=args.books_dir,
            test_csv=args.test_csv
        )
        
        # Run tests
        summary = tester.run_tests(k=args.k, max_tests=args.max_tests)
        
        # Print summary
        tester.print_summary(summary)
        
        # Save results to JSON
        output_file = Path('test_results.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nResults saved to {output_file}", flush=True)

    if __name__ == "__main__":
        main()
    '''

    from datetime import datetime

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_txt_path = out_dir / f"live_results_{timestamp}.txt"
    output_json_path = out_dir / f"live_results_{timestamp}.json"

    print(f"📄 Writing readable output to: {output_txt_path}", flush=True)
    print(f"📊 Writing JSON output to: {output_json_path}", flush=True)

    output_txt = open(output_txt_path, "w", encoding="utf-8")
    
    # Write header
    output_txt.write("="*80 + "\n")
    output_txt.write("NARRATIVE CONSISTENCY TEST RESULTS\n")
    output_txt.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    output_txt.write("="*80 + "\n\n")
    output_txt.flush()



    # --- NEW MAIN: process rows from train.csv and save per-row results ---
    from datetime import datetime
    train_csv = Path(__file__).parent.parent / "train.csv"
    if not train_csv.exists():
        print(f"Error: train.csv not found at {train_csv}")
        sys.exit(1)

    tester = NarrativeConsistencyTester()

    print(f"Processing rows from {train_csv} using NarrativeConsistencyTester...", flush=True)

    all_results = []
    with open(train_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, 1):
            test_case = {
                'id': row.get('id', ''),
                'book_name': row.get('book_name', ''),
                'char': row.get('char', ''),
                'content': row.get('content', ''),
                'label': row.get('label', '')
            }

            print(f"\n{'='*80}", flush=True)
            print(f"Row {row_idx} (ID: {test_case['id']}) - Book: {test_case['book_name']} - Char: {test_case['char']}", flush=True)
            
            # Write to formatted output
            output_txt.write(f"\n{'='*80}\n")
            output_txt.write(f"ROW {row_idx}\n")
            output_txt.write(f"{'='*80}\n")
            output_txt.write(f"ID: {test_case['id']}\n")
            output_txt.write(f"Book: {test_case['book_name']}\n")
            output_txt.write(f"Character: {test_case['char']}\n")
            output_txt.write(f"Expected Label: {test_case['label']}\n")
            output_txt.write(f"Backstory:\n{test_case['content']}\n")
            output_txt.write(f"\n{'-'*80}\n")
            output_txt.flush()

            result = tester.process_test_case(test_case, k=10, output_file=output_txt)

            all_results.append({'row': test_case, 'result': result})

    # Save aggregated results
    out_dir = Path(__file__).parent / 'output'
    out_dir.mkdir(exist_ok=True)
    
    with open(output_json_path, 'w', encoding='utf-8') as wf:
        json.dump(all_results, wf, indent=2, ensure_ascii=False, default=str)
    
    # Write summary to text file
    output_txt.write(f"\n\n{'='*80}\n")
    output_txt.write("SUMMARY\n")
    output_txt.write(f"{'='*80}\n")
    output_txt.write(f"Total rows processed: {len(all_results)}\n")
    
    successful = sum(1 for r in all_results if r['result'].get('success', False))
    failed = len(all_results) - successful
    output_txt.write(f"Successful: {successful}\n")
    output_txt.write(f"Failed: {failed}\n")
    
    # Count predictions
    correct = sum(1 for r in all_results if r['result'].get('success', False) and r['result'].get('label_match', False))
    total_valid = sum(1 for r in all_results if r['result'].get('success', False))
    accuracy = (correct / total_valid * 100) if total_valid > 0 else 0
    
    output_txt.write(f"Correct Predictions: {correct}/{total_valid}\n")
    output_txt.write(f"Accuracy: {accuracy:.2f}%\n")
    output_txt.write(f"{'='*80}\n")
    output_txt.flush()

    print(f"\nProcessing complete.", flush=True)
    print(f"Readable results saved to: {output_txt_path}", flush=True)
    print(f"JSON results saved to: {output_json_path}", flush=True)

    output_txt.close()
    print(f"✅ All results saved successfully", flush=True)



if __name__ == "__main__":
    main()

