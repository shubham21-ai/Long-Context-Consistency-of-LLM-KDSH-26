"""
Step 1: Simple helper to build narrative state from novel text.

This module provides a basic function to:
- Load novel text
- Process it through BDH model
- Build persistent sigma state matrix
"""

import os
import torch
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path

# Import BDH model
import sys
base_dir = Path(__file__).parent.parent
sys.path.insert(0, str(base_dir / "core"))
from bdh import BDH, BDHConfig


# Global model instance
_BDH_MODEL = None
_BDH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_bdh_model(config: Optional[BDHConfig] = None, model_path: Optional[str] = None) -> BDH:
    """Get or initialize BDH model instance."""
    global _BDH_MODEL
    
    if _BDH_MODEL is None:
        if config is None:
            config = BDHConfig()
        
        # Use higher hebbian_eta during ingestion to "burn" facts deeper into sigma matrix
        _BDH_MODEL = BDH(config, enable_hebbian=True, hebbian_eta=0.05).to(_BDH_DEVICE)
        _BDH_MODEL.eval()  # Set to inference mode for Hebbian learning
        
        # Load pretrained weights if available
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=_BDH_DEVICE)
            if 'model_state_dict' in checkpoint:
                _BDH_MODEL.load_state_dict(checkpoint['model_state_dict'])
            else:
                _BDH_MODEL.load_state_dict(checkpoint)
            print(f"✓ Loaded BDH model from {model_path}")
        else:
            print("⚠️  Using randomly initialized BDH model (no pretrained weights)")
    
    return _BDH_MODEL


def _text_to_tokens(text: str, vocab_size: int = 256) -> torch.Tensor:
    """
    Convert text to token indices (byte-level encoding).
    
    Args:
        text: Input text string
        vocab_size: Vocabulary size (256 for byte-level)
        
    Returns:
        Token indices tensor of shape (1, T)
    """
    # Byte-level encoding: convert string to bytes, then to integers
    text_bytes = bytearray(text, 'utf-8')
    tokens = torch.tensor([min(b, vocab_size - 1) for b in text_bytes], dtype=torch.long)
    return tokens.unsqueeze(0)  # Add batch dimension


def build_narrative_state_from_text(
    text: str,
    chunk_size: int = 512,
    model_path: Optional[str] = None,
    config: Optional[BDHConfig] = None
) -> Dict[str, Any]:
    """
    Step 1: Build narrative state (sigma matrix) from novel text.
    
    This function:
    1. Takes novel text as input
    2. Processes it through BDH model in chunks
    3. Updates sigma matrix using Hebbian learning
    4. Returns serialized state
    
    Args:
        text: Full novel text or text chunk
        chunk_size: Maximum tokens per chunk (default: 512)
        model_path: Optional path to pretrained BDH model
        config: Optional BDH configuration
        
    Returns:
        Dictionary with:
        - sigma: Serialized sigma matrix (numpy array as list)
        - config: BDH configuration
        - stats: Processing statistics
    """
    model = _get_bdh_model(config=config, model_path=model_path)
    
    # Reset sigma state for new narrative
    model.sigma.zero_()
    
    # Convert text to tokens
    tokens = _text_to_tokens(text)
    
    # Split into chunks if too long
    if tokens.size(1) > chunk_size:
        chunks = [tokens[:, i:i+chunk_size] for i in range(0, tokens.size(1), chunk_size)]
    else:
        chunks = [tokens]
    
    # Process each chunk through model
    total_updates = 0
    print(f"  Processing {len(chunks)} chunks...", flush=True)
    for i, chunk in enumerate(chunks):
        chunk = chunk.to(_BDH_DEVICE)
        with torch.no_grad():
            result = model(chunk, update_hebbian=True)
            # Check if Hebbian update occurred
            if isinstance(result, tuple) and len(result) >= 3:
                _, _, hebbian_update = result
                if hebbian_update:
                    total_updates += 1
                    # Print progress every 50 chunks or at start/end
                    if i == 0 or i == len(chunks) - 1 or (i + 1) % 50 == 0:
                        print(f"    Chunk {i+1}/{len(chunks)} (sigma norm: {hebbian_update.get('sigma_norm', 0):.4f})", flush=True)
    
    # Get final sigma state
    sigma_state = model.get_sigma_state()
    sigma_numpy = sigma_state['sigma'].numpy()
    
    return {
        'sigma': sigma_numpy.tolist(),  # Convert to list for JSON serialization
        'config': sigma_state['config'],
        'stats': {
            'processed_chunks': len(chunks),
            'total_updates': total_updates,
            'text_length': len(text)
        },
        'sigma_norm': float(torch.tensor(sigma_numpy).norm().item()),
        'sigma_shape': list(sigma_numpy.shape)
    }


if __name__ == "__main__":
    # Test with sample text
    test_text = "Raj is a character in the story. Raj is married to Preeti. They live together."
    print("Testing narrative state builder...")
    state = build_narrative_state_from_text(test_text)
    print(f"✓ Built state with sigma norm: {state['sigma_norm']:.4f}")
    print(f"  Processed {state['stats']['processed_chunks']} chunks")

