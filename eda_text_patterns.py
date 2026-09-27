"""
Phase 1: Deep Text & Noise Pattern Analysis Script
Samples ground-truth pairs and non-match pairs to evaluate string similarities,
noise variations, legal suffix frequencies, postal codes, and potential blocking keys.
"""

import os
import sys
import json
import re
from collections import Counter
import polars as pl
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

def clean_tokens(text):
    if not text:
        return []
    # Lowercase and split on non-alphanumeric (keep unicode word chars)
    return [t for t in re.findall(r'\w+', text.lower()) if t]

def jaccard(toks1, toks2):
    s1, s2 = set(toks1), set(toks2)
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)

# Common legal suffixes across US, India, France
LEGAL_SUFFIXES = {
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'ltd', 'limited',
    'pvt', 'private', 'co', 'company', 'sarl', 'sa', 'sas', 'eurl', 'gmbh',
    'enterprise', 'enterprises', 'services', 'solutions', 'technologies', 'group'
}

def remove_legal_suffixes(tokens):
    return [t for t in tokens if t not in LEGAL_SUFFIXES]

# Regex for postal codes:
# US: 5 digits (\b\d{5}\b)
# India: 6 digits (\b\d{6}\b)
# France: 5 digits (\b\d{5}\b)
PIN_RE = re.compile(r'\b(\d{5,6})\b')

def extract_pin(text):
    if not text:
        return None
    m = PIN_RE.findall(text)
    return m[-1] if m else None

def audit_text_and_noise(num_samples=50000):
    print("\n--- LOADING SAMPLES FOR TEXT & NOISE AUDIT ---")
    
    # Load S1
    s1_df = pl.read_csv(os.path.join(TRAIN_DIR, "train_source1.tsv"), separator="\t")
    s1_dict = {
        row["entity_id"]: (row["business_name"], row["business_address"], row["country"])
        for row in s1_df.slice(0, 100000).iter_rows(named=True)
    }
    
    # Load S2 and S3 for these S1 IDs
    s2_df = pl.read_csv(os.path.join(TRAIN_DIR, "train_source2.tsv"), separator="\t")
    s2_dict = {
        row["entity_id"]: (row["business_name"], row["business_address"], row["country"])
        for row in s2_df.slice(0, 250000).iter_rows(named=True)
    }
    
    s3_df = pl.read_csv(os.path.join(TRAIN_DIR, "train_source3.tsv"), separator="\t")
    s3_dict = {
        row["entity_id"]: (row["business_name"], row["business_address"], row["country"])
        for row in s3_df.slice(0, 250000).iter_rows(named=True)
    }
    
    gt_df = pl.read_csv(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), separator="\t").slice(0, 100000)
    
    # Collect true pairs where both sides are in our slice
    true_pairs = []
    for row in gt_df.iter_rows(named=True):
        s1_id = row["source1_entity_id"]
        if s1_id not in s1_dict:
            continue
        m_str = row["matched_entity_ids"]
        if not m_str:
            continue
        for tid in m_str.split(","):
            tid = tid.strip()
            target_data = s2_dict.get(tid) or s3_dict.get(tid)
            if target_data:
                true_pairs.append((s1_dict[s1_id], target_data))
                if len(true_pairs) >= num_samples:
                    break
        if len(true_pairs) >= num_samples:
            break
            
    print(f"Collected {len(true_pairs):,} true match pairs for detailed similarity analysis.")
    
    # Metrics on true pairs
    exact_raw_name = 0
    exact_lower_name = 0
    exact_norm_name = 0
    exact_raw_addr = 0
    exact_lower_addr = 0
    
    name_jaccards = []
    name_norm_jaccards = []
    addr_jaccards = []
    
    pin_both_present = 0
    pin_matches = 0
    
    has_non_ascii_name = 0
    has_non_ascii_addr = 0
    
    sample_variations = []
    
    for (s1_name, s1_addr, s1_c), (t_name, t_addr, t_c) in true_pairs:
        s1_name = s1_name or ""
        t_name = t_name or ""
        s1_addr = s1_addr or ""
        t_addr = t_addr or ""
        
        # Check non-ascii
        if any(ord(c) > 127 for c in s1_name) or any(ord(c) > 127 for c in t_name):
            has_non_ascii_name += 1
        if any(ord(c) > 127 for c in s1_addr) or any(ord(c) > 127 for c in t_addr):
            has_non_ascii_addr += 1
            
        # Exact checks
        if s1_name == t_name:
            exact_raw_name += 1
        if s1_name.lower().strip() == t_name.lower().strip():
            exact_lower_name += 1
            
        s1_n_toks = clean_tokens(s1_name)
        t_n_toks = clean_tokens(t_name)
        
        s1_n_clean = remove_legal_suffixes(s1_n_toks)
        t_n_clean = remove_legal_suffixes(t_n_toks)
        
        if s1_n_clean == t_n_clean and len(s1_n_clean) > 0:
            exact_norm_name += 1
            
        name_jaccards.append(jaccard(s1_n_toks, t_n_toks))
        name_norm_jaccards.append(jaccard(s1_n_clean, t_n_clean))
        
        # Address checks
        if s1_addr == t_addr:
            exact_raw_addr += 1
        if s1_addr.lower().strip() == t_addr.lower().strip():
            exact_lower_addr += 1
            
        s1_a_toks = clean_tokens(s1_addr)
        t_a_toks = clean_tokens(t_addr)
        addr_jaccards.append(jaccard(s1_a_toks, t_a_toks))
        
        # PIN / Postal code check
        p1 = extract_pin(s1_addr)
        p2 = extract_pin(t_addr)
        if p1 and p2:
            pin_both_present += 1
            if p1 == p2:
                pin_matches += 1
                
        if len(sample_variations) < 10 and s1_name.lower().strip() != t_name.lower().strip():
            sample_variations.append({
                "country": s1_c,
                "s1_name": s1_name,
                "t_name": t_name,
                "s1_addr": s1_addr,
                "t_addr": t_addr
            })

    N = len(true_pairs)
    print("\n--- RESULTS ON TRUE MATCH PAIRS ---")
    print(f"Exact raw name match: {exact_raw_name:,} / {N:,} ({exact_raw_name / N * 100:.2f}%)")
    print(f"Exact lower name match: {exact_lower_name:,} / {N:,} ({exact_lower_name / N * 100:.2f}%)")
    print(f"Exact suffix-cleaned name match: {exact_norm_name:,} / {N:,} ({exact_norm_name / N * 100:.2f}%)")
    print(f"Mean Name Token Jaccard: {np.mean(name_jaccards):.4f} (Median: {np.median(name_jaccards):.4f})")
    print(f"Mean Norm-Name Token Jaccard: {np.mean(name_norm_jaccards):.4f} (Median: {np.median(name_norm_jaccards):.4f})")
    print(f"Name Jaccard >= 0.5: {sum(1 for j in name_jaccards if j >= 0.5) / N * 100:.2f}%")
    print(f"Name Jaccard >= 0.2: {sum(1 for j in name_jaccards if j >= 0.2) / N * 100:.2f}%")
    
    print(f"\nExact raw address match: {exact_raw_addr:,} / {N:,} ({exact_raw_addr / N * 100:.2f}%)")
    print(f"Exact lower address match: {exact_lower_addr:,} / {N:,} ({exact_lower_addr / N * 100:.2f}%)")
    print(f"Mean Address Token Jaccard: {np.mean(addr_jaccards):.4f} (Median: {np.median(addr_jaccards):.4f})")
    
    print(f"\nPostal code in both records: {pin_both_present:,} / {N:,} ({pin_both_present / N * 100:.2f}%)")
    if pin_both_present > 0:
        print(f"Postal code match when both present: {pin_matches:,} / {pin_both_present:,} ({pin_matches / pin_both_present * 100:.2f}%)")
        
    print(f"\nNon-ASCII characters in names: {has_non_ascii_name:,} / {N:,} ({has_non_ascii_name / N * 100:.2f}%)")
    print(f"Non-ASCII characters in addresses: {has_non_ascii_addr:,} / {N:,} ({has_non_ascii_addr / N * 100:.2f}%)")
    
    print("\nSample True Match Variations:")
    for i, ex in enumerate(sample_variations[:5], 1):
        print(f"\nExample {i} ({ex['country']}):")
        print(f"  S1: '{ex['s1_name']}' | '{ex['s1_addr']}'")
        print(f"  T : '{ex['t_name']}' | '{ex['t_addr']}'")
        
    results = {
        "num_evaluated_pairs": N,
        "exact_raw_name_pct": exact_raw_name / N * 100,
        "exact_lower_name_pct": exact_lower_name / N * 100,
        "exact_norm_name_pct": exact_norm_name / N * 100,
        "mean_name_jaccard": float(np.mean(name_jaccards)),
        "median_name_jaccard": float(np.median(name_jaccards)),
        "name_jaccard_ge_05_pct": sum(1 for j in name_jaccards if j >= 0.5) / N * 100,
        "name_jaccard_ge_02_pct": sum(1 for j in name_jaccards if j >= 0.2) / N * 100,
        "exact_raw_addr_pct": exact_raw_addr / N * 100,
        "exact_lower_addr_pct": exact_lower_addr / N * 100,
        "mean_addr_jaccard": float(np.mean(addr_jaccards)),
        "median_addr_jaccard": float(np.median(addr_jaccards)),
        "pin_both_present_pct": pin_both_present / N * 100,
        "pin_match_pct_when_present": (pin_matches / pin_both_present * 100) if pin_both_present else 0.0,
        "has_non_ascii_name_pct": has_non_ascii_name / N * 100,
        "has_non_ascii_addr_pct": has_non_ascii_addr / N * 100,
        "sample_variations": sample_variations
    }
    
    with open("eda_noise_patterns.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nSaved eda_noise_patterns.json successfully.")

if __name__ == "__main__":
    audit_text_and_noise()
