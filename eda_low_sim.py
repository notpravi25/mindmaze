"""
Phase 1: Audit of Low Name-Similarity True Matches
Examines true match pairs where name token overlap is low to identify noise causes
(e.g., abbreviations, acronyms, transliterations, drastic noise).
"""

import os
import sys
import json
import re
import polars as pl

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

TRAIN_DIR = "dataset/train"

def clean_tokens(text):
    if not text:
        return set()
    return set(re.findall(r'\w+', text.lower()))

def main():
    s1_df = pl.read_csv(os.path.join(TRAIN_DIR, "train_source1.tsv"), separator="\t", n_rows=200000)
    s1_dict = {
        row["entity_id"]: (row["business_name"], row["business_address"], row["country"])
        for row in s1_df.iter_rows(named=True)
    }
    
    s2_df = pl.read_csv(os.path.join(TRAIN_DIR, "train_source2.tsv"), separator="\t", n_rows=500000)
    s2_dict = {
        row["entity_id"]: (row["business_name"], row["business_address"], row["country"])
        for row in s2_df.iter_rows(named=True)
    }
    
    s3_df = pl.read_csv(os.path.join(TRAIN_DIR, "train_source3.tsv"), separator="\t", n_rows=500000)
    s3_dict = {
        row["entity_id"]: (row["business_name"], row["business_address"], row["country"])
        for row in s3_df.iter_rows(named=True)
    }
    
    gt_df = pl.read_csv(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), separator="\t", n_rows=200000)
    
    low_sim_examples = []
    
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
                s1_name, s1_addr, s1_c = s1_dict[s1_id]
                t_name, t_addr, t_c = target_data
                toks1 = clean_tokens(s1_name)
                toks2 = clean_tokens(t_name)
                jacc = len(toks1 & toks2) / len(toks1 | toks2) if (toks1 | toks2) else 0.0
                if jacc < 0.3:
                    low_sim_examples.append({
                        "country": s1_c,
                        "s1_name": s1_name,
                        "t_name": t_name,
                        "jaccard": jacc,
                        "s1_addr": s1_addr,
                        "t_addr": t_addr
                    })
                    if len(low_sim_examples) >= 30:
                        break
        if len(low_sim_examples) >= 30:
            break
            
    print(f"Found {len(low_sim_examples)} examples of low-name-similarity true matches (Jaccard < 0.3):")
    for i, ex in enumerate(low_sim_examples[:15], 1):
        print(f"\n--- Low Sim Example {i} ({ex['country']}) --- Jaccard: {ex['jaccard']:.2f}")
        print(f"  S1 Name: {ex['s1_name']}")
        print(f"  T  Name: {ex['t_name']}")
        print(f"  S1 Addr: {ex['s1_addr']}")
        print(f"  T  Addr: {ex['t_addr']}")

if __name__ == "__main__":
    main()
