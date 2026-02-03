#!/usr/bin/env python3
"""
Compute Query-to-Anchor Similarity for SCOPE Router

This script computes semantic similarity between query questions and anchor questions
using Qwen3-Embedding-0.6B model. The output is a JSON file mapping each query to
its top-K most similar anchors.

Usage:
    python compute_similarity.py --query_file queries.json --output similarity.json
    python compute_similarity.py --query_file queries.json --dataset ood --output similarity.json

Requirements:
    - sentence-transformers>=2.7.0
    - transformers>=4.51.0
    - datasets
    - numpy
    - tqdm
"""

import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Optional

# Configuration
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
TOP_K = 10
BATCH_SIZE = 8
MAX_TEXT_LENGTH = 2000  # Truncate long texts

# HuggingFace dataset names
HF_DATASET_ID = "Cooolder/SCOPE-60K-final"
HF_DATASET_OOD = "Cooolder/SCOPE-60K-OOD-final"


def load_anchor_data(dataset_type: str = "id") -> List[Dict]:
    """
    Load anchor data from HuggingFace dataset.
    
    Args:
        dataset_type: 'id' for in-distribution (13 models), 'ood' for out-of-distribution (5 models)
    
    Returns:
        List of anchor records with 'id', 'prompt', 'category', 'gt', etc.
    """
    from datasets import load_dataset
    
    dataset_name = HF_DATASET_OOD if dataset_type == "ood" else HF_DATASET_ID
    print(f"Loading anchor data from {dataset_name}...")
    
    dataset = load_dataset(dataset_name, split="anchor")
    
    # Convert to list of dicts and deduplicate by question id
    seen_ids = set()
    anchor_data = []
    
    for item in dataset:
        qid = item['id']
        if qid not in seen_ids:
            seen_ids.add(qid)
            anchor_data.append({
                'id': qid,
                'prompt': item['prompt'],
                'category': item.get('category', ''),
                'gt': item.get('gt', ''),
                'source': item.get('source', item.get('category', '')),
            })
    
    print(f"Loaded {len(anchor_data)} unique anchor questions")
    return anchor_data


def load_query_data(query_file: str) -> List[Dict]:
    """
    Load query data from a JSON file.
    
    Expected format:
    [
        {"id": "q1", "prompt": "What is ...?", ...},
        {"id": "q2", "prompt": "How does ...?", ...}
    ]
    
    Args:
        query_file: Path to JSON file containing queries
    
    Returns:
        List of query records
    """
    print(f"Loading query data from {query_file}...")
    
    with open(query_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Ensure each item has required fields
    for item in data:
        if 'prompt' not in item and 'question' in item:
            item['prompt'] = item['question']
        if 'id' not in item:
            item['id'] = f"query_{data.index(item)}"
    
    print(f"Loaded {len(data)} query questions")
    return data


def load_embedding_model(use_gpu: bool = True):
    """
    Load Qwen3-Embedding-0.6B model.
    
    Args:
        use_gpu: Whether to use GPU if available
    
    Returns:
        SentenceTransformer model
    """
    from sentence_transformers import SentenceTransformer
    import torch
    
    print("="*70)
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    print("="*70)
    
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        # Try with flash attention for better performance
        model = SentenceTransformer(
            EMBEDDING_MODEL,
            model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
            tokenizer_kwargs={"padding_side": "left"},
        )
    except Exception:
        # Fallback to standard loading
        model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    
    print("Model loaded successfully")
    return model


def generate_embeddings(model, data: List[Dict], data_name: str = "data") -> np.ndarray:
    """
    Generate embeddings for a list of text prompts.
    
    Args:
        model: SentenceTransformer model
        data: List of dicts with 'prompt' field
        data_name: Name for progress display
    
    Returns:
        numpy array of embeddings (N x embedding_dim)
    """
    print()
    print("="*70)
    print(f"Generating embeddings for {data_name} ({len(data)} samples)")
    print("="*70)
    
    # Extract and truncate prompts
    prompts = [item['prompt'] for item in data]
    
    # Count and truncate long texts
    long_texts = sum(1 for p in prompts if len(p) > MAX_TEXT_LENGTH)
    if long_texts > 0:
        print(f"  Truncating {long_texts} texts longer than {MAX_TEXT_LENGTH} characters")
    
    truncated_prompts = [p[:MAX_TEXT_LENGTH] if len(p) > MAX_TEXT_LENGTH else p for p in prompts]
    
    # Generate embeddings in batches
    all_embeddings = []
    
    for i in tqdm(range(0, len(truncated_prompts), BATCH_SIZE), desc=f"{data_name}"):
        batch = truncated_prompts[i:i+BATCH_SIZE]
        embeddings = model.encode(batch, show_progress_bar=False)
        all_embeddings.append(embeddings)
    
    embeddings_array = np.vstack(all_embeddings)
    print(f"Embeddings shape: {embeddings_array.shape}")
    
    return embeddings_array


def compute_similarities(
    query_embeddings: np.ndarray,
    anchor_embeddings: np.ndarray,
    query_data: List[Dict],
    anchor_data: List[Dict],
    top_k: int = TOP_K
) -> List[Dict]:
    """
    Compute cosine similarity and find top-K similar anchors for each query.
    
    Args:
        query_embeddings: Query embeddings (N_query x dim)
        anchor_embeddings: Anchor embeddings (N_anchor x dim)
        query_data: Query metadata
        anchor_data: Anchor metadata
        top_k: Number of similar anchors to return
    
    Returns:
        List of similarity records
    """
    print()
    print("="*70)
    print(f"Computing Top-{top_k} similarities")
    print("="*70)
    
    # Normalize embeddings for cosine similarity
    query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    anchor_norms = np.linalg.norm(anchor_embeddings, axis=1, keepdims=True)
    
    query_normalized = query_embeddings / (query_norms + 1e-9)
    anchor_normalized = anchor_embeddings / (anchor_norms + 1e-9)
    
    results = []
    
    for i in tqdm(range(len(query_data)), desc="Computing similarities"):
        query_item = query_data[i]
        query_emb = query_normalized[i:i+1]  # shape: (1, dim)
        
        # Compute similarity with all anchors
        similarities = np.dot(query_emb, anchor_normalized.T)[0]  # shape: (N_anchor,)
        
        # Get top-K indices
        top_k_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Build result record
        similar_anchors = []
        for rank, anchor_idx in enumerate(top_k_indices, start=1):
            anchor_item = anchor_data[anchor_idx]
            similar_anchors.append({
                "rank": rank,
                "anchor_id": anchor_item['id'],
                "anchor_question": anchor_item['prompt'][:200] + "..." if len(anchor_item['prompt']) > 200 else anchor_item['prompt'],
                "anchor_category": anchor_item.get('category', ''),
                "anchor_gt": anchor_item.get('gt', ''),
                "anchor_source": anchor_item.get('source', ''),
                "similarity": float(similarities[anchor_idx])
            })
        
        results.append({
            "router_id": query_item['id'],
            "router_question": query_item['prompt'][:200] + "..." if len(query_item['prompt']) > 200 else query_item['prompt'],
            "router_category": query_item.get('category', ''),
            "router_gt": query_item.get('gt', ''),
            "similar_anchors": similar_anchors
        })
    
    print(f"Computed similarities for {len(results)} queries")
    return results


def save_results(results: List[Dict], output_file: str):
    """Save similarity results to JSON file."""
    print()
    print("="*70)
    print("Saving results")
    print("="*70)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Saved to: {output_file}")
    
    # Print statistics
    print()
    print("Statistics:")
    print(f"  Total queries: {len(results)}")
    print(f"  Anchors per query: {TOP_K}")
    
    if results:
        avg_top1_sim = np.mean([r['similar_anchors'][0]['similarity'] for r in results])
        avg_top10_sim = np.mean([r['similar_anchors'][-1]['similarity'] for r in results])
        print(f"  Average Top-1 similarity: {avg_top1_sim:.4f}")
        print(f"  Average Top-{TOP_K} similarity: {avg_top10_sim:.4f}")


def save_embeddings(embeddings: np.ndarray, metadata: List[Dict], output_dir: str, prefix: str):
    """
    Save embeddings and metadata for later reuse.
    
    Args:
        embeddings: Numpy array of embeddings
        metadata: List of metadata dicts
        output_dir: Directory to save files
        prefix: Prefix for filenames (e.g., 'anchor', 'query')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save embeddings
    emb_file = os.path.join(output_dir, f"{prefix}_embeddings.npy")
    np.save(emb_file, embeddings)
    print(f"Saved embeddings to: {emb_file}")
    
    # Save metadata
    meta_file = os.path.join(output_dir, f"{prefix}_metadata.json")
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Saved metadata to: {meta_file}")


def load_cached_embeddings(cache_dir: str, prefix: str):
    """
    Load cached embeddings if available.
    
    Returns:
        (embeddings, metadata) tuple, or (None, None) if not cached
    """
    emb_file = os.path.join(cache_dir, f"{prefix}_embeddings.npy")
    meta_file = os.path.join(cache_dir, f"{prefix}_metadata.json")
    
    if os.path.exists(emb_file) and os.path.exists(meta_file):
        print(f"Loading cached {prefix} embeddings from {cache_dir}")
        embeddings = np.load(emb_file)
        with open(meta_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        return embeddings, metadata
    
    return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Compute query-to-anchor similarity for SCOPE Router"
    )
    parser.add_argument(
        "--query_file", "-q",
        type=str,
        required=True,
        help="Path to JSON file containing query questions"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="similarity_index.json",
        help="Output file path for similarity results"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        choices=["id", "ood"],
        default="id",
        help="Dataset type: 'id' for in-distribution, 'ood' for out-of-distribution"
    )
    parser.add_argument(
        "--top_k", "-k",
        type=int,
        default=TOP_K,
        help=f"Number of similar anchors to retrieve (default: {TOP_K})"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory to cache/load anchor embeddings"
    )
    parser.add_argument(
        "--save_embeddings",
        action="store_true",
        help="Save computed embeddings for later reuse"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU usage even if GPU is available"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("SCOPE Router: Query-to-Anchor Similarity Computation")
    print("="*70)
    print()
    
    # Load model
    model = load_embedding_model(use_gpu=not args.cpu)
    
    # Try to load cached anchor embeddings
    anchor_embeddings = None
    anchor_data = None
    
    if args.cache_dir:
        anchor_embeddings, anchor_data = load_cached_embeddings(args.cache_dir, "anchor")
    
    # Load/generate anchor embeddings
    if anchor_embeddings is None:
        anchor_data = load_anchor_data(args.dataset)
        anchor_embeddings = generate_embeddings(model, anchor_data, "anchor")
        
        if args.save_embeddings and args.cache_dir:
            save_embeddings(anchor_embeddings, anchor_data, args.cache_dir, "anchor")
    
    # Load query data and generate embeddings
    query_data = load_query_data(args.query_file)
    query_embeddings = generate_embeddings(model, query_data, "query")
    
    if args.save_embeddings and args.cache_dir:
        save_embeddings(query_embeddings, query_data, args.cache_dir, "query")
    
    # Compute similarities
    global TOP_K
    TOP_K = args.top_k
    results = compute_similarities(
        query_embeddings, anchor_embeddings,
        query_data, anchor_data,
        top_k=args.top_k
    )
    
    # Save results
    save_results(results, args.output)
    
    print()
    print("="*70)
    print("Done!")
    print("="*70)


if __name__ == "__main__":
    main()
