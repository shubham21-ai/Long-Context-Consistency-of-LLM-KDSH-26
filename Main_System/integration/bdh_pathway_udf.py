"""
Pathway UDFs for Stateful BDH Inference with Hebbian Learning.

This module provides Pathway UDFs for:
1. build_narrative_state: Ingests novel chunks and builds persistent sigma state
2. verify_causal_consistency: Measures synaptic tension to verify claim consistency
"""

import os
import pickle
import json
import torch
import numpy as np
from typing import List, Dict, Optional, Any
import pathway as pw

# Import BDH model
import sys
base_dir = Path(__file__).parent.parent
sys.path.insert(0, str(base_dir / "core"))
sys.path.insert(0, str(base_dir / "utils"))
from bdh import BDH, BDHConfig
from bdh_narrative_builder import _text_to_tokens


# Global BDH model instance (loaded once, reused)
_BDH_MODEL = None
_BDH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_bdh_model(config: Optional[BDHConfig] = None, model_path: Optional[str] = None) -> BDH:
    """Get or initialize BDH model instance."""
    global _BDH_MODEL
    
    if _BDH_MODEL is None:
        if config is None:
            config = BDHConfig()
        
        _BDH_MODEL = BDH(config, enable_hebbian=True, hebbian_eta=0.01).to(_BDH_DEVICE)
        _BDH_MODEL.eval()  # Set to inference mode
        
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


def _process_text_chunk(model: BDH, text_chunk: str, chunk_size: int = 512) -> Dict[str, Any]:
    """
    Process a single text chunk through BDH model to update sigma state.
    
    Args:
        model: BDH model instance
        text_chunk: Text chunk to process
        chunk_size: Maximum chunk size in tokens
        
    Returns:
        Dictionary with processing statistics
    """
    tokens = _text_to_tokens(text_chunk)
    
    # Split into chunks if too long
    if tokens.size(1) > chunk_size:
        chunks = [tokens[:, i:i+chunk_size] for i in range(0, tokens.size(1), chunk_size)]
    else:
        chunks = [tokens]
    
    total_updates = 0
    for chunk in chunks:
        chunk = chunk.to(_BDH_DEVICE)
        with torch.no_grad():
            result = model(chunk, update_hebbian=True)
            if isinstance(result, tuple) and len(result) >= 3:
                _, _, hebbian_update = result
                if hebbian_update:
                    total_updates += 1
    
    return {
        'processed': True,
        'chunk_length': tokens.size(1),
        'num_chunks': len(chunks),
        'hebbian_updates': total_updates
    }


@pw.udf
def build_narrative_state(text_stream: str) -> pw.Json:
    """
    Pathway UDF: Build Global Narrative State (S_N) from text stream.
    
    Ingests novel chunks, runs forward passes through BDH model to "absorb" facts
    into the sigma matrix using Inference-Time Hebbian Learning.
    
    Args:
        text_stream: Text content from novel (can be chunked or full text)
        
    Returns:
        Serialized sigma state as JSON with:
        - sigma: Serialized sigma matrix (base64 encoded pickle)
        - config: BDH configuration
        - stats: Processing statistics
    """
    model = _get_bdh_model()
    
    # Reset sigma state for new narrative
    nh = model.config.n_head
    N = model.config.mlp_internal_dim_multiplier * model.config.n_embd // nh
    model.sigma.zero_()
    
    # Process text stream
    stats = _process_text_chunk(model, text_stream)
    
    # Get final sigma state
    sigma_state = model.get_sigma_state()
    
    # Serialize sigma matrix (convert to numpy for JSON serialization)
    sigma_numpy = sigma_state['sigma'].numpy()
    
    # Create serializable state
    state_dict = {
        'sigma': sigma_numpy.tolist(),  # Convert to list for JSON
        'config': sigma_state['config'],
        'stats': stats,
        'sigma_norm': float(torch.tensor(sigma_numpy).norm().item()),
        'sigma_shape': list(sigma_numpy.shape)
    }
    
    return pw.Json(state_dict)


@pw.udf
def verify_causal_consistency(claim: str, sigma_state: pw.Json) -> pw.Json:
    """
    Pathway UDF: Verify causal consistency of a claim against saved sigma state.
    
    Probes the BDH model with the claim using the saved sigma_state.
    Measures 'Synaptic Tension' (inhibitory signals) between the claim and the state.
    
    Logic: If tension > threshold, return Verdict=0 (Inconsistent).
    
    Args:
        claim: Claim text to verify
        sigma_state: Serialized sigma state from build_narrative_state
        
    Returns:
        JSON with:
        - verdict: 0 (Inconsistent) or 1 (Consistent)
        - tension: Synaptic tension value
        - threshold: Tension threshold used
        - reasoning: Detailed reasoning with neuron-level information
    """
    model = _get_bdh_model()
    
    # Deserialize sigma state
    state_dict = sigma_state.value
    sigma_array = np.array(state_dict['sigma'], dtype=np.float32)
    sigma_tensor = torch.from_numpy(sigma_array).to(_BDH_DEVICE)
    
    # Load sigma state into model
    model.load_sigma_state({'sigma': sigma_tensor})
    
    # Process claim to get activations
    claim_tokens = _text_to_tokens(claim).to(_BDH_DEVICE)
    with torch.no_grad():
        claim_presynaptic, claim_postsynaptic = model.extract_activations(claim_tokens)
    
    # Compute synaptic tension
    tension_result = model.compute_synaptic_tension(
        claim_presynaptic,
        sigma_state={'sigma': sigma_tensor}
    )
    
    # Threshold for consistency (tunable parameter)
    TENSION_THRESHOLD = 0.5  # Higher = stricter consistency check
    
    total_tension = tension_result['total_tension']
    mean_tension = tension_result['mean_tension']
    
    # Verdict: 0 = Inconsistent, 1 = Consistent
    verdict = 0 if mean_tension > TENSION_THRESHOLD else 1
    
    # Build detailed reasoning with neuron-level information
    reasoning_parts = [
        f"Total synaptic tension: {total_tension:.4f}",
        f"Mean tension per head: {mean_tension:.4f}",
        f"Threshold: {TENSION_THRESHOLD}",
        f"Verdict: {'INCONSISTENT' if verdict == 0 else 'CONSISTENT'}"
    ]
    
    # Add top conflicting neurons information
    if tension_result.get('top_conflicting_neurons'):
        reasoning_parts.append("\nTop conflicting neurons:")
        for head_info in tension_result['top_conflicting_neurons']:
            if head_info['neuron_ids']:
                reasoning_parts.append(
                    f"  Head {head_info['head']}: Neurons {head_info['neuron_ids'][:3]} "
                    f"(tensions: {[f'{t:.3f}' for t in head_info['tensions'][:3]]})"
                )
    
    reasoning = "\n".join(reasoning_parts)
    
    return pw.Json({
        'verdict': verdict,
        'tension': mean_tension,
        'total_tension': total_tension,
        'threshold': TENSION_THRESHOLD,
        'reasoning': reasoning,
        'per_head_tension': tension_result['per_head_tension'],
        'top_conflicting_neurons': tension_result.get('top_conflicting_neurons', [])
    })


# Helper function for batch processing (non-UDF, can be used in Python code)
def build_narrative_state_batch(text_chunks: List[str], model_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Build narrative state from multiple text chunks (non-UDF version for Python use).
    
    Args:
        text_chunks: List of text chunks from novel
        model_path: Optional path to pretrained BDH model
        
    Returns:
        Serialized sigma state dictionary
    """
    model = _get_bdh_model(model_path=model_path)
    
    # Reset sigma state
    nh = model.config.n_head
    N = model.config.mlp_internal_dim_multiplier * model.config.n_embd // nh
    model.sigma.zero_()
    
    # Process all chunks
    total_stats = {'processed_chunks': 0, 'total_updates': 0}
    for chunk in text_chunks:
        stats = _process_text_chunk(model, chunk)
        total_stats['processed_chunks'] += 1
        total_stats['total_updates'] += stats.get('hebbian_updates', 0)
    
    # Get final state
    sigma_state = model.get_sigma_state()
    sigma_numpy = sigma_state['sigma'].numpy()
    
    return {
        'sigma': sigma_numpy.tolist(),
        'config': sigma_state['config'],
        'stats': total_stats,
        'sigma_norm': float(torch.tensor(sigma_numpy).norm().item()),
        'sigma_shape': list(sigma_numpy.shape)
    }


def verify_causal_consistency_batch(
    claims: List[str],
    sigma_state: Dict[str, Any],
    model_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Verify multiple claims against sigma state (non-UDF version for Python use).
    
    Args:
        claims: List of claim texts to verify
        sigma_state: Serialized sigma state dictionary
        model_path: Optional path to pretrained BDH model
        
    Returns:
        List of verification results
    """
    model = _get_bdh_model(model_path=model_path)
    
    # Deserialize sigma state
    sigma_array = np.array(sigma_state['sigma'], dtype=np.float32)
    sigma_tensor = torch.from_numpy(sigma_array).to(_BDH_DEVICE)
    model.load_sigma_state({'sigma': sigma_tensor})
    
    results = []
    TENSION_THRESHOLD = 0.5
    
    for claim in claims:
        claim_tokens = _text_to_tokens(claim).to(_BDH_DEVICE)
        with torch.no_grad():
            claim_presynaptic, _ = model.extract_activations(claim_tokens)
        
        tension_result = model.compute_synaptic_tension(
            claim_presynaptic,
            sigma_state={'sigma': sigma_tensor}
        )
        
        mean_tension = tension_result['mean_tension']
        verdict = 0 if mean_tension > TENSION_THRESHOLD else 1
        
        results.append({
            'claim': claim,
            'verdict': verdict,
            'tension': mean_tension,
            'total_tension': tension_result['total_tension'],
            'threshold': TENSION_THRESHOLD,
            'reasoning': f"Tension {mean_tension:.4f} {'exceeds' if verdict == 0 else 'below'} threshold {TENSION_THRESHOLD}",
            'per_head_tension': tension_result['per_head_tension']
        })
    
    return results

