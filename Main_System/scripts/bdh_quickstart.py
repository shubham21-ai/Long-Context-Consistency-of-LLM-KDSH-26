"""
BDH Claim Verification - Real Books Training & Testing
======================================================
Trains BDH model on actual novels from Books folder and tests with real claims.
"""

import os
import sys
import torch
import numpy as np
import csv
from pathlib import Path

# Add parent directories to path
base_dir = Path(__file__).parent.parent
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(base_dir / "core"))
sys.path.insert(0, str(base_dir / "utils"))

from bdh import BDH, BDHConfig
from bdh_narrative_builder import _text_to_tokens
from csv_loader import load_csv_file

def load_books(books_dir: Path) -> list[str]:
    """Load all novels from Books directory."""
    books = []
    for book_file in books_dir.glob("*.txt"):
        print(f"  Loading {book_file.name}...")
        with open(book_file, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            books.append(text)
            print(f"    ✓ Loaded {len(text):,} characters")
    return books

def train_model_on_books(books: list[str], config: BDHConfig, epochs: int = 50, 
                         model_save_path: str = None) -> BDH:
    """Train BDH model on real novels."""
    print(f"\n🔥 Training BDH model on {len(books)} novels...")
    print(f"   Config: {config.n_layer} layers, {config.n_embd} dim, {config.n_head} heads")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    
    model = BDH(config, enable_hebbian=False).to(device)
    model.train()
    
    # Combine all books into one training corpus
    combined_text = "\n\n".join(books)
    print(f"   Total training text: {len(combined_text):,} characters")
    
    # Prepare training data
    tokens = _text_to_tokens(combined_text).squeeze(0)
    seq_len = 64 if config.n_embd == 128 else 128  # Shorter for small models
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    
    # Create batches
    batches = []
    for i in range(0, len(tokens) - seq_len - 1, seq_len // 2):
        chunk = tokens[i:i+seq_len+1]
        if len(chunk) == seq_len + 1:
            batches.append((chunk[:-1], chunk[1:]))
    
    print(f"   Created {len(batches)} training batches")
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        
        # Shuffle batches
        np.random.shuffle(batches)
        
        # Process in mini-batches
        for i in range(0, len(batches), batch_size):
            batch = batches[i:i+batch_size]
            if len(batch) < batch_size:
                continue
            
            xs = torch.stack([b[0] for b in batch]).to(device)
            ys = torch.stack([b[1] for b in batch]).to(device)
            
            optimizer.zero_grad()
            logits, loss = model(xs, targets=ys)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        if n_batches > 0:
            avg_loss = total_loss / n_batches
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{epochs}: loss = {avg_loss:.4f}")
    
    print("✓ Training complete!")
    
    # Save model if path provided
    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'epochs': epochs
        }, model_save_path)
        print(f"✓ Model saved to {model_save_path}")
    
    return model

def build_state_from_narrative(model: BDH, narrative: str, eta: float = 0.2, chunk_size: int = 256):
    """Build Hebbian state from narrative (handles long texts)."""
    print("\n🧠 Building narrative state...")
    
    device = next(model.parameters()).device
    
    # Enable Hebbian learning
    model.enable_hebbian = True
    model.hebbian_eta = eta
    model.sigma.zero_()
    model.eval()
    
    # Process narrative in chunks
    tokens = _text_to_tokens(narrative).squeeze(0)
    
    if len(tokens) > chunk_size:
        chunks = [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]
        print(f"  Processing {len(chunks)} chunks...")
    else:
        chunks = [tokens]
    
    for i, chunk in enumerate(chunks):
        chunk = chunk.unsqueeze(0).to(device)
        with torch.no_grad():
            result = model(chunk, update_hebbian=True)
            if isinstance(result, tuple) and len(result) >= 3:
                hebbian_update = result[2]
                if hebbian_update and (i == 0 or i == len(chunks) - 1):
                    print(f"  Chunk {i+1}/{len(chunks)}: sigma_norm = {hebbian_update['sigma_norm']:.4f}")
    
    state = model.get_sigma_state()
    print(f"✓ Narrative state built! Final sigma norm: {state['sigma'].norm().item():.4f}")
    
    return state

def verify_claim(model: BDH, claim: str, sigma_state: dict, baseline_tension: float = None):
    """Verify a claim against the narrative state."""
    device = next(model.parameters()).device
    
    # Load state
    sigma_tensor = sigma_state['sigma'].to(device)
    model.load_sigma_state({'sigma': sigma_tensor})
    model.eval()
    
    # Process claim
    tokens = _text_to_tokens(claim).to(device)
    
    with torch.no_grad():
        pre, _ = model.extract_activations(tokens)
        tension_result = model.compute_synaptic_tension(
            pre, 
            sigma_state={'sigma': sigma_tensor}
        )
    
    tension = tension_result['mean_tension']
    
    # Use dynamic threshold based on baseline
    if baseline_tension is not None:
        threshold = baseline_tension * 1.5  # 50% higher than baseline
        verdict = 0 if tension > threshold else 1
    else:
        threshold = 0.1
        verdict = 0 if tension > threshold else 1
    
    return {
        'tension': tension,
        'threshold': threshold,
        'verdict': verdict,
        'verdict_str': 'CONSISTENT' if verdict == 1 else 'INCONSISTENT'
    }

def load_test_cases(test_csv_path: Path, train_csv_path: Path = None, max_cases: int = None):
    """Load test cases from test.csv and match labels from train.csv by ID."""
    print(f"\n📋 Loading test cases from {test_csv_path.name}...")
    
    try:
        # Load test cases (has: id, book_name, char, content)
        test_rows = load_csv_file(
            file_path=str(test_csv_path),
            csv_filename=None,
            credentials_path=None
        )
        print(f"  ✓ Loaded {len(test_rows)} rows from test.csv")
        
        # Load labels from train.csv (has: id, book_name, char, content, label)
        label_map = {}
        if train_csv_path and train_csv_path.exists():
            print(f"  Loading labels from {train_csv_path.name}...")
            train_rows = load_csv_file(
                file_path=str(train_csv_path),
                csv_filename=None,
                credentials_path=None
            )
            for train_row in train_rows:
                test_id = train_row.get('id', '').strip()
                label = train_row.get('label', '').strip().lower()
                if test_id and label:
                    label_map[test_id] = label
            print(f"  ✓ Loaded {len(label_map)} labels from train.csv")
        else:
            print(f"  ⚠️  train.csv not found at {train_csv_path}")
        
        # Combine test cases with labels
        test_cases = []
        for row in test_rows[:max_cases] if max_cases else test_rows:
            test_id = row.get('id', '').strip()
            label = label_map.get(test_id, 'unknown')
            
            test_cases.append({
                'id': test_id,
                'book_name': row.get('book_name', '').strip(),
                'char': row.get('char', '').strip(),
                'content': row.get('content', '').strip(),  # Backstory claim to verify
                'label': label  # consistent or contradict from train.csv
            })
        
        # Count how many have labels
        labeled_count = sum(1 for tc in test_cases if tc['label'] != 'unknown')
        print(f"  ✓ Created {len(test_cases)} test cases ({labeled_count} with labels)")
        
        return test_cases
    except Exception as e:
        print(f"  ⚠️  Could not load CSV: {e}")
        import traceback
        traceback.print_exc()
        return []

def main():
    """Main training and testing pipeline with real books."""
    print("=" * 80)
    print("BDH CLAIM VERIFICATION - REAL BOOKS TRAINING & TESTING")
    print("=" * 80)
    
    # Paths
    base_dir = Path(__file__).parent.parent.parent
    books_dir = base_dir / "Books"
    test_csv = base_dir / "test.csv"
    train_csv = base_dir / "train.csv"  # For labels
    model_save_path = Path(__file__).parent / "models" / "bdh_trained.pt"
    
    # ========================================================================
    # STEP 1: Load real novels
    # ========================================================================
    
    print("\n📚 Step 1: Loading novels from Books folder...")
    if not books_dir.exists():
        print(f"  ✗ Books directory not found: {books_dir}")
        return
    
    books = load_books(books_dir)
    if not books:
        print("  ✗ No books found!")
        return
    
    # ========================================================================
    # STEP 2: Train model (or load if exists)
    # ========================================================================
    
    print("\n🎯 Step 2: Training/loading model...")
    
    # Model configuration
    # For MacBook Air 8GB RAM: Use smaller config (n_layer=4, n_embd=128)
    # For Google Colab/GPU: Use full config (n_layer=6, n_embd=256)
    USE_SMALL_MODEL = True  # Set False for Colab/GPU training
    
    if USE_SMALL_MODEL:
        print("  ⚠️  Using SMALL model config (optimized for 8GB RAM)")
        config = BDHConfig(
            n_layer=4,  # Reduced for memory efficiency
            n_embd=128,  # Reduced for memory efficiency
            n_head=4,
            mlp_internal_dim_multiplier=64,  # Reduced
            vocab_size=256,
            dropout=0.1
        )
    else:
        print("  ✓ Using FULL model config (for GPU/Colab)")
        config = BDHConfig(
            n_layer=6,  # Full size for better performance
            n_embd=256,
            n_head=4,
            mlp_internal_dim_multiplier=128,
            vocab_size=256,
            dropout=0.1
        )
    
    if model_save_path.exists():
        print(f"  Loading saved model from {model_save_path}...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_save_path, map_location=device)
        model = BDH(checkpoint['config'], enable_hebbian=False).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("  ✓ Model loaded!")
    else:
        print("  Training new model (this will take a while)...")
        # Training parameters
        # For MacBook Air: epochs=20-30, batch_size=4
        # For Colab/GPU: epochs=50-100, batch_size=8
        epochs = 30 if USE_SMALL_MODEL else 50
        batch_size = 4 if USE_SMALL_MODEL else 8
        
        print(f"  Training parameters: epochs={epochs}, batch_size={batch_size}")
        
        model = train_model_on_books(
            books=books,
            config=config,
            epochs=epochs,
            batch_size=batch_size,
            model_save_path=str(model_save_path)
        )
    
    # ========================================================================
    # STEP 3: Load test cases and build narrative states
    # ========================================================================
    
    print("\n📖 Step 3: Processing test cases...")
    
    test_cases = load_test_cases(test_csv, train_csv, max_cases=5)  # Start with 5 for testing
    
    if not test_cases:
        print("  ⚠️  No test cases, using demo...")
        # Fallback demo
        narrative = "Raj is a software engineer. Raj is married to Preeti. They live in Palo Alto."
        sigma_state = build_state_from_narrative(model, narrative, eta=0.2)
        
        print("\n🔍 Step 4: Verifying demo claims...")
        baseline_claim = "Raj is married to Preeti"
        baseline_result = verify_claim(model, baseline_claim, sigma_state)
        baseline_tension = baseline_result['tension']
        
        test_claims = [
            ("Raj is married to Preeti", True),
            ("Raj is married to Simran", False),
        ]
        
        results = []
        for claim, expected_consistent in test_claims:
            result = verify_claim(model, claim, sigma_state, baseline_tension)
            expected_verdict = 1 if expected_consistent else 0
            is_correct = result['verdict'] == expected_verdict
            results.append(is_correct)
            print(f"  {'✓' if is_correct else '✗'} '{claim}' → {result['verdict_str']} "
                  f"(tension: {result['tension']:.4f})")
        
        accuracy = sum(results) / len(results) * 100 if results else 0
        print(f"\n✓ Demo accuracy: {accuracy:.1f}%")
        return
    
    # Process each test case
    print(f"\n🔍 Step 4: Verifying {len(test_cases)} test cases...")
    print("=" * 80)
    
    all_results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] Test Case ID: {test_case['id']}")
        print(f"  Book: {test_case['book_name']}")
        print(f"  Character: {test_case['char']}")
        print(f"  Backstory: {test_case['content'][:100]}...")
        print(f"  Expected: {'CONSISTENT' if test_case['label'] == 'consistent' else 'INCONSISTENT'}")
        
        # Find the book text for this test case
        book_name = test_case['book_name']
        book_text = None
        for book_file in books_dir.glob("*.txt"):
            if book_name.lower() in book_file.name.lower():
                with open(book_file, 'r', encoding='utf-8', errors='ignore') as f:
                    book_text = f.read()
                break
        
        if not book_text:
            print("  ⚠️  Book not found, skipping...")
            continue
        
        # Build narrative state from the book
        print(f"  Building state from book ({len(book_text):,} chars)...")
        sigma_state = build_state_from_narrative(model, book_text, eta=0.2, chunk_size=512)
        
        # The backstory claim to verify
        backstory_claim = test_case['content']
        
        # Get baseline tension from a simple character mention (for calibration)
        # Use character name as baseline
        char_name = test_case['char']
        if char_name:
            baseline_claim = f"{char_name} is a character in the story"
            baseline_result = verify_claim(model, baseline_claim, sigma_state)
            baseline_tension = baseline_result['tension']
        else:
            # Fallback: use first sentence of backstory as baseline
            baseline_claim = backstory_claim.split('.')[0] if '.' in backstory_claim else backstory_claim[:50]
            baseline_result = verify_claim(model, baseline_claim, sigma_state)
            baseline_tension = baseline_result['tension']
        
        print(f"  Baseline tension: {baseline_tension:.4f} (from: '{baseline_claim[:60]}...')")
        
        # Verify the backstory claim against the book's narrative state
        result = verify_claim(model, backstory_claim, sigma_state, baseline_tension)
        
        # Determine expected verdict
        # train.csv uses: 'consistent' or 'contradict'
        label = test_case['label'].lower()
        if label in ['consistent', 'contradict']:
            expected_verdict = 1 if label == 'consistent' else 0
            is_correct = result['verdict'] == expected_verdict
            expected_str = label.upper()
        else:
            # No label available, just report verdict
            expected_verdict = None
            is_correct = None
            expected_str = "UNKNOWN"
        
        status = "✓" if is_correct else ("?" if is_correct is None else "✗")
        color = "🟢" if result['verdict'] == 1 else "🔴"
        
        print(f"  {status} Verdict: {result['verdict_str']} {color}")
        print(f"     Expected: {expected_str}")
        print(f"     Tension: {result['tension']:.4f} (threshold: {result['threshold']:.4f})")
        print(f"     Claim: '{backstory_claim[:80]}...'")
        
        all_results.append({
            'test_id': test_case['id'],
            'expected': label,  # consistent or contradict
            'predicted': 'consistent' if result['verdict'] == 1 else 'contradict',
            'correct': is_correct,
            'tension': result['tension'],
            'claim': backstory_claim[:100]
        })
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    if all_results:
        accuracy = sum(1 for r in all_results if r['correct']) / len(all_results) * 100
        
        print("\n" + "=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)
        print(f"Accuracy: {accuracy:.1f}% ({sum(1 for r in all_results if r['correct'])}/{len(all_results)} correct)")
        print("\nDetailed Results:")
        for r in all_results:
            status = "✓" if r['correct'] else ("?" if r['correct'] is None else "✗")
            print(f"  {status} Test {r['test_id']}: Expected {r['expected']}, Got {r['predicted']} "
                  f"(tension: {r['tension']:.4f})")
            if r['claim']:
                print(f"      Claim: {r['claim']}")
        
        if accuracy >= 70:
            print("\n🎉 Excellent! The system is working well!")
        elif accuracy >= 50:
            print("\n⚠️  Partial success. Try:")
            print("  - Training for more epochs")
            print("  - Adjusting hebbian_eta (try 0.1 to 0.3)")
            print("  - Using more baseline claims")
        else:
            print("\n❌ Low accuracy. The model needs:")
            print("  - More training epochs (100+)")
            print("  - Better baseline calibration")
            print("  - More diverse training data")
    else:
        print("\n⚠️  No results to report.")

if __name__ == "__main__":
    main()

