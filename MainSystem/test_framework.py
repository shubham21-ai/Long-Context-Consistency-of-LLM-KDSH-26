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
from questions.question_generator import generate_questions_from_event
from rag import RAG
from evaluator import evaluate_consistency, apply_decision_rule


class NarrativeConsistencyTester:
    """Test framework for narrative consistency checking."""
    
    def __init__(self, books_dir: str = None, test_csv: str = None, 
                 model_llm: str = "gemini-2.0-flash", 
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
                    'label': row.get('label', '0')  # Expected label (1 = consistent, 0 = inconsistent)
                })
        return test_cases
    
    def process_test_case(self, test_case: Dict, k: int = 15) -> Dict:
        """
        Single-pass processing of a test case:
        retrieval + evaluation + decision + label comparison
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
            # Extract claims
            # ------------------------------------------------------------
            print("Extracting claims...", flush=True)
            claims = extract_claims(backstory, character)
            if not claims:
                raise ValueError("No claims extracted")

            # ------------------------------------------------------------
            # Generate questions
            # ------------------------------------------------------------
            questions = []
            for claim in claims:
                questions.extend(generate_questions_from_event(claim, character))
            if not questions:
                raise ValueError("No questions generated")

            # ------------------------------------------------------------
            # Retrieve + Evaluate (INLINE)
            # ------------------------------------------------------------
            evaluations = []
            rag_results = {}

            for i, q in enumerate(questions, 1):
                print(f"  [{i}/{len(questions)}] {q[:60]}...", flush=True)

                docs = rag.retrieve_data(q, k=k)
                chunks = [
                    d.page_content if hasattr(d, "page_content") else str(d)
                    for d in docs
                ]
                rag_results[q] = chunks

                eval_result = evaluate_consistency(q, chunks, backstory)
                evaluations.append(eval_result)

                print(
                    f"    → {eval_result.get('verdict', 'UNCERTAIN')}",
                    flush=True
                )

            # ------------------------------------------------------------
            # Decision rule
            # ------------------------------------------------------------
            decision = apply_decision_rule(evaluations)
            verdict = decision['verdict']

            predicted_label = '0' if verdict == 'INCONSISTENT' else '1'
            label_match = (predicted_label == expected_label)

            # ------------------------------------------------------------
            # PRINT FINAL ROW RESULT
            # ------------------------------------------------------------
            print(f"\n  FINAL VERDICT : {verdict}", flush=True)
            print(f"  EXPECTED      : {expected_label}", flush=True)
            print(f"  PREDICTED     : {predicted_label}", flush=True)
            print(f"  MATCH         : {'✓' if label_match else '✗'}", flush=True)
            print("-" * 80, flush=True)

            return {
                'test_case': test_case,
                'claims': claims,
                'questions': questions,
                'rag_results': rag_results,
                'evaluations': evaluations,
                'decision_rule': decision,
                'predicted_label': predicted_label,
                'expected_label': expected_label,
                'label_match': label_match,
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

            result = self.process_test_case(test_case, k=k)
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
        print(f"Inconsistent Evaluations: {summary.get('total_inconsistent', 0)}", flush=True)
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
            expected_text = "INCONSISTENT" if expected_label == '0' else "CONSISTENT"
            
            print(f"\nTest {i}: {test_case['book_name']} - {test_case['char']}", flush=True)
            print(f"  Expected: {expected_text} (label: {expected_label})", flush=True)
            
            if 'decision_rule' in result:
                dr = result['decision_rule']
                verdict = dr['verdict']
                confidence = dr['confidence']
                is_correct = (expected_label == '0' and verdict == 'INCONSISTENT') or \
                            (expected_label == '1' and verdict in ['CONSISTENT', 'LIKELY_CONSISTENT'])
                status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
                
                print(f"  Predicted: {verdict} (confidence: {confidence:.2f}) {status}", flush=True)
                print(f"  Reason: {dr['verdict_reason']}", flush=True)
                
                signals = dr.get('signals', {})
                print(f"  Evaluations: {signals.get('consistent_count', 0)} consistent, "
                      f"{signals.get('inconsistent_count', 0)} inconsistent, "
                      f"{signals.get('uncertain_count', 0)} uncertain", flush=True)
            
            if 'evaluations' in result:
                inconsistent_evals = [e for e in result['evaluations'] if e.get('verdict') == 'INCONSISTENT']
                if inconsistent_evals:
                    print(f"  ⚠️  INCONSISTENT EVALUATIONS: {len(inconsistent_evals)}", flush=True)
                    for eval_result in inconsistent_evals[:3]:  # Show first 3
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

            result = tester.process_test_case(test_case, k=10)

            # Attempt to make claims serializable
            if isinstance(result, dict) and result.get('success', False):
                serial_claims = []
                for c in result.get('claims', []):
                    try:
                        serial_claims.append({
                            'subject': getattr(c, 'subject', str(c)),
                            'relation': getattr(c, 'relation', ''),
                            'object': getattr(c, 'object', ''),
                            'time': getattr(c, 'time', '') if hasattr(c, 'time') else '',
                            'location': getattr(c, 'location', '') if hasattr(c, 'location') else ''
                        })
                    except Exception:
                        serial_claims.append(str(c))
                result['claims_serialized'] = serial_claims

            all_results.append({'row': test_case, 'result': result})

    # Save aggregated results
    out_dir = Path(__file__).parent / 'output'
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = out_dir / f'test_framework_train_results_{timestamp}.json'
    with open(out_file, 'w', encoding='utf-8') as wf:
        json.dump(all_results, wf, indent=2, ensure_ascii=False, default=str)

    print(f"\nProcessing complete. Results saved to: {out_file}", flush=True)


if __name__ == "__main__":
    main()

