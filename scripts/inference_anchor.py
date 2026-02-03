#!/usr/bin/env python3
"""
Anchor Set Inference for Custom Model Pool

When using a custom model pool, you must first run inference on the anchor set
to obtain historical performance data that the router needs.

This script:
1. Loads anchor questions from HuggingFace
2. Runs inference on each anchor question with each model in the custom pool
3. Saves results locally for use by the routing algorithm

Usage:
    python inference_anchor.py --model_pool custom_pool.txt --output data/anchor_results/
    python inference_anchor.py --model_pool custom_pool.txt --dataset ood

Requirements:
    - OpenRouter API key (set OPENROUTER_API_KEY environment variable)
    - Custom model pool file (one OpenRouter model ID per line)
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_pools import load_custom_pool, PRICING
from compute_similarity import HF_DATASET_ID, HF_DATASET_OOD

# OpenRouter API configuration
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


def load_anchor_questions(dataset_type: str = "id") -> List[Dict]:
    """
    Load unique anchor questions from HuggingFace.
    
    Returns:
        List of anchor questions with 'id', 'prompt', 'gt', 'category'
    """
    from datasets import load_dataset
    
    dataset_name = HF_DATASET_OOD if dataset_type == "ood" else HF_DATASET_ID
    print(f"Loading anchor questions from {dataset_name}...")
    
    dataset = load_dataset(dataset_name, split="anchor")
    
    # Deduplicate by question ID
    seen_ids = set()
    anchors = []
    
    for item in dataset:
        qid = item['id']
        if qid not in seen_ids:
            seen_ids.add(qid)
            anchors.append({
                'id': qid,
                'prompt': item['prompt'],
                'gt': item.get('gt', ''),
                'category': item.get('category', ''),
            })
    
    print(f"Loaded {len(anchors)} unique anchor questions")
    return anchors


def call_openrouter(
    prompt: str,
    model_id: str,
    api_key: str,
    max_tokens: int = 2048,
    temperature: float = 0.7
) -> Dict:
    """
    Call OpenRouter API for a single prompt.
    
    Returns:
        Dict with 'response', 'usage', 'error' fields
    """
    import requests
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/scope-router",
    }
    
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'response': data['choices'][0]['message']['content'],
                    'usage': data.get('usage', {}),
                    'error': None
                }
            elif response.status_code == 429:
                # Rate limited
                wait_time = RETRY_DELAY * (attempt + 1)
                print(f"  Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                return {
                    'response': None,
                    'usage': {},
                    'error': f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                return {
                    'response': None,
                    'usage': {},
                    'error': str(e)
                }
    
    return {'response': None, 'usage': {}, 'error': "Max retries exceeded"}


def extract_answer(response: str) -> str:
    """Extract answer from model response."""
    if not response:
        return ""
    
    # Look for common answer patterns
    import re
    
    # Pattern: "The answer is X" or "Answer: X"
    patterns = [
        r'[Tt]he answer is[:\s]*([A-E])',
        r'[Aa]nswer[:\s]*([A-E])',
        r'\b([A-E])\s*(?:is|would be)\s+(?:the\s+)?(?:correct|right|best)',
        r'(?:^|\n)\s*([A-E])\s*(?:\.|$)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1).upper()
    
    # Last resort: find any single letter A-E
    letters = re.findall(r'\b([A-E])\b', response)
    if letters:
        return letters[-1].upper()
    
    return ""


def check_correctness(extracted: str, ground_truth: str) -> bool:
    """Check if extracted answer matches ground truth."""
    if not extracted or not ground_truth:
        return False
    return extracted.upper() == ground_truth.upper()


def run_anchor_inference(
    anchors: List[Dict],
    model_id: str,
    api_key: str,
    output_dir: str,
    skip_existing: bool = True
) -> List[Dict]:
    """
    Run inference on all anchors for a single model.
    
    Returns:
        List of result dicts
    """
    # Create output directory for this model
    model_dir_name = model_id.replace('/', '_')
    model_output_dir = Path(output_dir) / model_dir_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    result_file = model_output_dir / "anchor_results.json"
    
    # Load existing results if any
    existing_results = {}
    if skip_existing and result_file.exists():
        with open(result_file, 'r') as f:
            existing = json.load(f)
            existing_results = {r['id']: r for r in existing}
        print(f"  Found {len(existing_results)} existing results")
    
    results = []
    errors = 0
    
    for anchor in tqdm(anchors, desc=f"  {model_id}"):
        qid = anchor['id']
        
        # Skip if already processed
        if qid in existing_results:
            results.append(existing_results[qid])
            continue
        
        # Call API
        api_result = call_openrouter(anchor['prompt'], model_id, api_key)
        
        if api_result['error']:
            errors += 1
            result = {
                'id': qid,
                'model': model_id,
                'response': None,
                'extracted_answer': '',
                'is_correct': False,
                'error': api_result['error'],
                'usage': {},
            }
        else:
            extracted = extract_answer(api_result['response'])
            is_correct = check_correctness(extracted, anchor['gt'])
            
            result = {
                'id': qid,
                'model': model_id,
                'response': api_result['response'],
                'extracted_answer': extracted,
                'ground_truth': anchor['gt'],
                'is_correct': is_correct,
                'error': None,
                'usage': api_result['usage'],
            }
        
        results.append(result)
        
        # Save incrementally
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Final save
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Compute stats
    correct = sum(1 for r in results if r.get('is_correct', False))
    print(f"  Accuracy: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")
    if errors:
        print(f"  Errors: {errors}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on anchor set for custom model pool"
    )
    parser.add_argument(
        "--model_pool", "-m",
        type=str,
        required=True,
        help="Path to custom model pool file (one OpenRouter ID per line)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/anchor_results",
        help="Output directory for anchor results"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        choices=["id", "ood"],
        default="id",
        help="Dataset type: 'id' or 'ood'"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip already processed questions"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of anchors to process (for testing)"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("Anchor Set Inference for Custom Model Pool")
    print("="*70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Get API key
    api_key = args.api_key or os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        print("Error: OpenRouter API key required.")
        print("Set OPENROUTER_API_KEY environment variable or use --api_key")
        return 1
    
    # Load custom model pool
    print(f"Loading model pool from {args.model_pool}...")
    models = load_custom_pool(args.model_pool)
    print(f"Models in pool: {len(models)}")
    for m in models:
        print(f"  - {m}")
    print()
    
    # Load anchor questions
    anchors = load_anchor_questions(args.dataset)
    
    if args.limit:
        anchors = anchors[:args.limit]
        print(f"Limited to {len(anchors)} anchors for testing")
    
    # Run inference for each model
    print("\n" + "="*70)
    print("Running inference")
    print("="*70)
    
    all_results = {}
    for model_id in models:
        print(f"\nModel: {model_id}")
        results = run_anchor_inference(
            anchors, model_id, api_key, args.output, args.skip_existing
        )
        all_results[model_id] = results
    
    # Save summary
    summary_file = Path(args.output) / "summary.json"
    summary = {
        'timestamp': datetime.now().isoformat(),
        'dataset': args.dataset,
        'num_anchors': len(anchors),
        'models': {},
    }
    
    for model_id, results in all_results.items():
        correct = sum(1 for r in results if r.get('is_correct', False))
        summary['models'][model_id] = {
            'accuracy': correct / len(results) if results else 0,
            'correct': correct,
            'total': len(results),
        }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    for model_id, stats in summary['models'].items():
        print(f"  {model_id}: {stats['accuracy']*100:.1f}% ({stats['correct']}/{stats['total']})")
    
    print(f"\n✅ Results saved to: {args.output}")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    exit(main())
