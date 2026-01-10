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
import os
import time
import requests
from pathlib import Path
from typing import Dict, List, Tuple

# Suppress tokenizers warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Ensure output is not buffered
try:
    sys.stdout.reconfigure(line_buffering=True)
except:
    pass

from backstory.claim_extractor import extract_claims
from questions.question_generator import generate_questions_from_event
from evaluator import evaluate_consistency, apply_decision_rule

# Import Pathway VectorStoreClient
pathway_server_path = Path(__file__).parent / "Pathway_code"
sys.path.insert(0, str(pathway_server_path))
try:
    from server import VectorStoreClient
except ImportError:
    # Fallback: VectorStoreClient not available, will use direct HTTP requests
    VectorStoreClient = None
    print("⚠️  VectorStoreClient not available, will use direct HTTP requests", flush=True)


class NarrativeConsistencyTester:
    """Test framework for narrative consistency checking."""
    
    def __init__(self, books_dir: str = None, test_csv: str = None,
                 pathway_host: str = "127.0.0.1", pathway_port: int = 8745):
        """
        Initialize the test framework.
        
        Args:
            books_dir: Directory containing novel text files (default: ../Books) - not used, kept for compatibility
            test_csv: Path to test CSV file with backstories (default: ../test.csv)
            pathway_host: Pathway server host (default: 127.0.0.1)
            pathway_port: Pathway server port (default: 8745)
        """
        # Set default paths relative to lohiya_code directory
        if books_dir is None:
            books_dir = Path(__file__).parent.parent / "Books"
        if test_csv is None:
            test_csv = Path(__file__).parent.parent / "test.csv"
        
        self.books_dir = Path(books_dir)
        self.test_csv = Path(test_csv)
        self.pathway_host = pathway_host
        self.pathway_port = pathway_port
        self.pathway_url = f"http://{pathway_host}:{pathway_port}"
        
        # Initialize Pathway client
        if VectorStoreClient:
            try:
                self.pathway_client = VectorStoreClient(host=pathway_host, port=pathway_port)
                print(f"✓ Connected to Pathway server at {self.pathway_url}", flush=True)
            except Exception as e:
                print(f"⚠️  Warning: Could not initialize Pathway client: {e}", flush=True)
                self.pathway_client = None
        else:
            self.pathway_client = None
    
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
    
    def query_pathway_server(self, question: str, k: int = 15, character: str = "") -> List[str]:
        """
        Query Pathway RAG server and retrieve relevant chunks.
        
        Args:
            question: Query question
            k: Number of chunks to retrieve
            character: Character name (for filtering, not used in Pathway)
            
        Returns:
            List of retrieved text chunks
        """
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Try using VectorStoreClient first
                if self.pathway_client:
                    results = self.pathway_client.query(question, k=k)
                    # Extract text from results (results are dicts with 'text' key)
                    chunks = []
                    for result in results:
                        if isinstance(result, dict):
                            chunks.append(result.get('text', str(result)))
                        else:
                            chunks.append(str(result))
                    return chunks[:k]
                
                # Fallback to direct HTTP request
                url = f"{self.pathway_url}/v1/retrieve"
                response = requests.post(
                    url,
                    json={"query": question, "k": k},
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                response.raise_for_status()
                results = response.json()
                
                # Extract text from results
                chunks = []
                for result in results:
                    if isinstance(result, dict):
                        chunks.append(result.get('text', ''))
                    else:
                        chunks.append(str(result))
                return chunks[:k]
                
            except (requests.exceptions.ConnectionError, requests.exceptions.RequestException) as e:
                if attempt < max_retries - 1:
                    print(f"    ⚠️  Connection attempt {attempt + 1} failed, retrying in {retry_delay}s...", flush=True)
                    time.sleep(retry_delay)
                else:
                    print(f"    ✗ ERROR: Failed to connect to Pathway server after {max_retries} attempts", flush=True)
                    raise ConnectionError(f"Could not connect to Pathway server at {self.pathway_url}: {e}")
            except Exception as e:
                print(f"    ✗ ERROR querying Pathway server: {e}", flush=True)
                raise
        
        return []
    
    def check_server_status(self) -> bool:
        """Check if Pathway server is running."""
        try:
            url = f"{self.pathway_url}/v1/statistics"
            response = requests.post(
                url,
                json={},
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            response.raise_for_status()
            stats = response.json()
            file_count = stats.get('file_count', 0)
            print(f"✓ Pathway server is running ({file_count} files indexed)", flush=True)
            return True
        except Exception as e:
            print(f"✗ Pathway server is not responding: {e}", flush=True)
            print(f"  Make sure the server is running:", flush=True)
            print(f"  cd Main_System/Pathway_code && python3 run-server-gdrive.py", flush=True)
            return False
    
    def process_test_case(self, test_case: Dict, k: int = 15) -> Dict:
        """
        Process a single test case: extract claims, generate questions, query story using Pathway RAG.
        
        Args:
            test_case: Dictionary with test case data
            k: Number of chunks to retrieve per question
            
        Returns:
            Dictionary with processing results
        """
        book_name = test_case['book_name']
        character = test_case['char']
        backstory = test_case['content']
        
        try:
            # Check server status
            if not self.check_server_status():
                raise ConnectionError("Pathway server is not running. Please start it first.")
            
            # Extract claims from backstory
            print("Extracting claims from backstory...", flush=True)
            claims = extract_claims(backstory, character)
            
            # Generate questions
            print(f"Generating questions from {len(claims)} claims...", flush=True)
            questions = []
            for idx, claim in enumerate(claims, 1):
                print(f"  Generating questions for claim {idx}/{len(claims)}...", flush=True)
                event_questions = generate_questions_from_event(claim, character)
                questions.extend(event_questions)
            
            # Query Pathway RAG server for each question
            print(f"\nQuerying Pathway RAG server (retrieving top {k} chunks per question)...", flush=True)
            rag_results = {}
            for i, question in enumerate(questions, 1):
                print(f"  Querying {i}/{len(questions)}: {question[:60]}...", flush=True)
                try:
                    retrieved_chunks = self.query_pathway_server(question, k=k, character=character)
                    rag_results[question] = retrieved_chunks
                    print(f"    ✓ Retrieved {len(retrieved_chunks)} chunks", flush=True)
                    if retrieved_chunks:
                        print(f"    Preview: {retrieved_chunks[0][:80]}...", flush=True)
                    else:
                        print(f"    ⚠️  WARNING: No chunks retrieved for this question!", flush=True)
                except Exception as e:
                    print(f"    ✗ ERROR: {str(e)}", flush=True)
                    rag_results[question] = []
            
            print("RAG queries completed", flush=True)
            
            result_dict = {
                'test_case': test_case,
                'claims': claims,
                'questions': questions,
                'rag_results': rag_results,
                'num_chunks': sum(len(chunks) for chunks in rag_results.values()),  # Total chunks retrieved
                'character': character,
                'success': True
            }
            
            print(f"  Processed {len(claims)} claims, generated {len(questions)} questions, retrieved chunks for all queries", flush=True)
            
            return result_dict
            
        except Exception as e:
            import traceback
            return {
                'test_case': test_case,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'success': False
            }
    
    def run_tests(self, k: int = 8, max_tests: int = None) -> Dict:
        """
        Run all tests and evaluate accuracy.
        
        Args:
            k: Number of chunks to retrieve per question
            max_tests: Maximum number of tests to run (None for all)
            
        Returns:
            Dictionary with test results and metrics
        """
        print("="*80)
        print("NARRATIVE CONSISTENCY TEST FRAMEWORK")
        print("="*80)
        
        # Load test data
        print(f"\nLoading test data from {self.test_csv}...", flush=True)
        test_cases = self.load_test_data()
        if max_tests:
            test_cases = test_cases[:max_tests]
        print(f"Loaded {len(test_cases)} test cases", flush=True)
        
        results = []
        total_consistent = 0
        total_inconsistent = 0
        total_uncertain = 0
        correct_predictions = 0
        total_predictions = 0
        
        for i, test_case in enumerate(test_cases, 1):
            try:
                print(f"\n{'='*80}", flush=True)
                print(f"Processing Test Case {i}/{len(test_cases)}", flush=True)
                print(f"Book: {test_case['book_name']}", flush=True)
                print(f"Character: {test_case['char']}", flush=True)
                print(f"Backstory: {test_case['content'][:100]}...", flush=True)
                print(f"{'='*80}", flush=True)
                
                # Process test case
                print(f"  → Starting test case processing...", flush=True)
                result = self.process_test_case(test_case, k)
                print(f"  → Test case processing completed. Success: {result.get('success', False)}", flush=True)
                
                # Debug: Check result structure
                try:
                    print(f"  → Result keys: {list(result.keys()) if isinstance(result, dict) else 'NOT_A_DICT'}", flush=True)
                    print(f"  → Questions present: {'questions' in result if isinstance(result, dict) else False}, Count: {len(result.get('questions', [])) if isinstance(result, dict) else 0}", flush=True)
                    print(f"  → RAG results present: {'rag_results' in result if isinstance(result, dict) else False}, Count: {len(result.get('rag_results', {})) if isinstance(result, dict) else 0}", flush=True)
                except Exception as debug_e:
                    print(f"  ⚠️  Debug check failed: {debug_e}", flush=True)
                
                if not isinstance(result, dict):
                    print(f"  ✗ ERROR: process_test_case returned non-dict: {type(result)}", flush=True)
                    results.append({
                        'test_case': test_case,
                        'error': f'Invalid return type: {type(result)}',
                        'success': False
                    })
                    continue
                
                if not result.get('success', False):
                    error_msg = result.get('error', 'Unknown error')
                    print(f"  ✗ ERROR: {error_msg}", flush=True)
                    if 'traceback' in result:
                        print(f"  Traceback: {result['traceback'][:500]}...", flush=True)
                    results.append(result)
                    continue
                
                # Verify we have questions and rag_results
                if 'questions' not in result or not result.get('questions'):
                    print(f"  ⚠️  WARNING: No questions generated! Questions: {result.get('questions', 'MISSING')}", flush=True)
                    results.append(result)
                    continue
                    
                if 'rag_results' not in result or not result.get('rag_results'):
                    print(f"  ⚠️  WARNING: No RAG results! RAG results: {result.get('rag_results', 'MISSING')}", flush=True)
                    results.append(result)
                    continue
                
                # SINGLE EVALUATION: Compare retrieved answers with backstory facts
                print(f"\n  → Starting evaluation phase...", flush=True)
                print(f"Evaluating {len(result['questions'])} questions against backstory facts...", flush=True)
                evaluations = []
                
                for idx, question in enumerate(result['questions'], 1):
                    retrieved_chunks = result['rag_results'].get(question, [])
                    
                    # Single evaluation: Compare retrieved answer with backstory facts
                    print(f"  Evaluating {idx}/{len(result['questions'])}: {question[:60]}...", flush=True)
                    try:
                        eval_result = evaluate_consistency(
                            question,
                            retrieved_chunks,
                            test_case['content']  # Full backstory as facts
                        )
                        
                        evaluations.append(eval_result)
                        
                        verdict = eval_result.get('verdict', 'UNCERTAIN')
                        answer = eval_result.get('answer', 'NOT_MENTIONED')
                        answer_str = answer[:100] if isinstance(answer, str) and len(answer) > 100 else str(answer)
                        print(f"    ✓ Verdict: {verdict}, Answer: {answer_str[:60]}...", flush=True)
                    except Exception as e:
                        import traceback
                        print(f"    ✗ ERROR evaluating question {idx}: {str(e)}", flush=True)
                        print(f"    Traceback: {traceback.format_exc()[:200]}...", flush=True)
                        # Add error evaluation
                        evaluations.append({
                            'verdict': 'UNCERTAIN',
                            'answer': 'ERROR',
                            'consistent': False,
                            'confidence': 0.0,
                            'reasoning': f'Evaluation error: {str(e)[:100]}'
                        })
                
                # Apply simple decision rule
                print(f"\nApplying decision rule...", flush=True)
                decision_result = apply_decision_rule(evaluations)
                
                print(f"  Final Verdict: {decision_result['verdict']}", flush=True)
                print(f"  Reason: {decision_result['verdict_reason']}", flush=True)
                print(f"  Confidence: {decision_result['confidence']:.2f}", flush=True)
                
                result['evaluations'] = evaluations
                result['decision_rule'] = decision_result
                result['expected_label'] = test_case.get('label', '0')  # Store expected label for accuracy
                results.append(result)
                
            except Exception as outer_e:
                import traceback
                print(f"\n  ✗✗✗ UNHANDLED EXCEPTION in test case loop: {str(outer_e)}", flush=True)
                print(f"  Full traceback:\n{traceback.format_exc()}", flush=True)
                results.append({
                    'test_case': test_case if 'test_case' in locals() else {},
                    'error': f'Unhandled exception: {str(outer_e)}',
                    'traceback': traceback.format_exc(),
                    'success': False
                })
                continue
        
        # Calculate accuracy based on decision rule verdicts vs expected labels
        for result in results:
            if result.get('success', False) and 'decision_rule' in result:
                expected_label = result.get('expected_label', '0')
                verdict = result['decision_rule']['verdict']
                
                # Map verdict to binary label: INCONSISTENT -> 0, CONSISTENT/LIKELY_CONSISTENT -> 1
                predicted_label = '0' if verdict in ['INCONSISTENT'] else '1'
                
                if expected_label == predicted_label:
                    correct_predictions += 1
                total_predictions += 1
        
        # Count evaluation verdicts across all results
        for result in results:
            if result.get('success', False) and 'evaluations' in result:
                for eval_result in result['evaluations']:
                    verdict = eval_result.get('verdict', 'UNCERTAIN').upper()
                    if verdict == 'CONSISTENT':
                        total_consistent += 1
                    elif verdict == 'INCONSISTENT':
                        total_inconsistent += 1
                    else:
                        total_uncertain += 1
        
        accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
        
        print(f"\n{'='*80}", flush=True)
        print(f"COMPLETED ALL TEST CASES", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"Total tests: {len(test_cases)}", flush=True)
        print(f"Successful tests: {sum(1 for r in results if r.get('success', False))}", flush=True)
        print(f"Total evaluations: {total_consistent + total_inconsistent + total_uncertain}", flush=True)
        print(f"Correct predictions: {correct_predictions}/{total_predictions}", flush=True)
        print(f"Accuracy: {accuracy:.2f}%", flush=True)
        
        summary = {
            'total_tests': len(test_cases),
            'successful_tests': sum(1 for r in results if r.get('success', False)),
            'total_consistent': total_consistent,
            'total_inconsistent': total_inconsistent,
            'total_uncertain': total_uncertain,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions,
            'accuracy': accuracy,
            'results': results
        }
        
        return summary
    
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
    """Main entry point for test framework."""
    import argparse
    
    # Ensure unbuffered output
    import sys
    sys.stdout = sys.__stdout__  # Reset to ensure clean output
    os.environ['PYTHONUNBUFFERED'] = '1'
    
    parser = argparse.ArgumentParser(description='Test narrative consistency framework')
    parser.add_argument('--books-dir', default=None, help='Directory containing books (default: ../Books) - not used with Pathway')
    parser.add_argument('--test-csv', default=None, help='Path to test CSV file (default: ../test.csv)')
    parser.add_argument('--k', type=int, default=10, help='Number of chunks to retrieve')
    parser.add_argument('--max-tests', type=int, default=None, help='Maximum number of tests to run')
    parser.add_argument('--pathway-host', default='127.0.0.1', help='Pathway server host (default: 127.0.0.1)')
    parser.add_argument('--pathway-port', type=int, default=8745, help='Pathway server port (default: 8745)')
    
    args = parser.parse_args()
    
    # Create tester
    tester = NarrativeConsistencyTester(
        books_dir=args.books_dir,
        test_csv=args.test_csv,
        pathway_host=args.pathway_host,
        pathway_port=args.pathway_port
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

