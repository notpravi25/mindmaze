"""
Phase 1: Deep Ground Truth & Noise Pattern Audit Script
Analyzes match distribution, cross-source links, country consistency, and noise patterns.
"""

import os
import sys
import json
import re
from collections import Counter, defaultdict
import polars as pl

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

DATASET_DIR = "dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")

def audit_ground_truth():
    print("\n--- AUDITING TRAIN GROUND TRUTH ---")
    gt_path = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")
    gt_df = pl.read_csv(gt_path, separator="\t", infer_schema_length=0)
    
    total_s1 = gt_df.height
    print(f"Total S1 entities in ground truth: {total_s1:,}")
    
    # Analyze matched_entity_ids
    # Count matches per S1
    match_counts = []
    s2_match_counts = []
    s3_match_counts = []
    singleton_count = 0
    multi_match_count = 0
    single_match_count = 0
    
    match_source_breakdown = Counter() # (num_s2, num_s3)
    duplicates_in_gt = 0
    
    # Streaming iteration over rows
    # Convert to python iterator of tuples for granular analysis
    count_freq = Counter()
    
    for row in gt_df.iter_rows(named=True):
        m_str = row["matched_entity_ids"]
        if m_str is None or m_str.strip() == "":
            singleton_count += 1
            count_freq[0] += 1
            match_source_breakdown[(0, 0)] += 1
            continue
            
        m_list = [x.strip() for x in m_str.split(",") if x.strip()]
        num_m = len(m_list)
        count_freq[num_m] += 1
        
        if len(set(m_list)) != num_m:
            duplicates_in_gt += 1
            
        if num_m == 1:
            single_match_count += 1
        else:
            multi_match_count += 1
            
        s2_count = sum(1 for x in m_list if x.startswith("S2-"))
        s3_count = sum(1 for x in m_list if x.startswith("S3-"))
        match_source_breakdown[(s2_count, s3_count)] += 1

    print(f"Singletons (0 matches): {singleton_count:,} ({singleton_count / total_s1 * 100:.2f}%)")
    print(f"Single match (1 match): {single_match_count:,} ({single_match_count / total_s1 * 100:.2f}%)")
    print(f"Multi-match (>=2 matches): {multi_match_count:,} ({multi_match_count / total_s1 * 100:.2f}%)")
    print(f"Duplicate IDs inside a match list: {duplicates_in_gt}")
    
    print("\nMatch count frequency distribution:")
    for k in sorted(count_freq.keys()):
        print(f"  Matches = {k}: {count_freq[k]:,} S1 entities ({count_freq[k] / total_s1 * 100:.3f}%)")
        
    print("\nTop 15 (num_S2, num_S3) patterns per S1:")
    for (s2_c, s3_c), count in match_source_breakdown.most_common(15):
        print(f"  S2: {s2_c}, S3: {s3_c} -> {count:,} S1 entities ({count / total_s1 * 100:.2f}%)")

    return {
        "total_s1": total_s1,
        "singletons": singleton_count,
        "single_match": single_match_count,
        "multi_match": multi_match_count,
        "count_freq": dict(count_freq),
        "top_source_breakdown": {f"S2_{k[0]}_S3_{k[1]}": v for k, v in match_source_breakdown.most_common(10)}
    }

def audit_countries():
    print("\n--- AUDITING COUNTRIES ACROSS SOURCES ---")
    results = {}
    for name, path in [
        ("train_s1", os.path.join(TRAIN_DIR, "train_source1.tsv")),
        ("train_s2", os.path.join(TRAIN_DIR, "train_source2.tsv")),
        ("train_s3", os.path.join(TRAIN_DIR, "train_source3.tsv")),
        ("test_s1", os.path.join(TEST_DIR, "test_source1.tsv")),
        ("test_s2", os.path.join(TEST_DIR, "test_source2.tsv")),
        ("test_s3", os.path.join(TEST_DIR, "test_source3.tsv")),
    ]:
        q = pl.scan_csv(path, separator="\t", infer_schema_length=0)
        c_dist = q.group_by("country").len().collect().sort("len", descending=True)
        print(f"\n{name} Country Distribution:")
        print(c_dist)
        results[name] = {r[0]: int(r[1]) for r in c_dist.iter_rows()}
    return results

def audit_cross_country_matches(sample_size=100000):
    print("\n--- CHECKING COUNTRY CONSISTENCY IN GROUND TRUTH ---")
    # Read S1 country
    s1_countries = {}
    print("Loading S1 countries...")
    s1_df = pl.read_csv(os.path.join(TRAIN_DIR, "train_source1.tsv"), separator="\t", columns=["entity_id", "country"])
    s1_map = dict(zip(s1_df["entity_id"], s1_df["country"]))
    
    print("Loading S2 countries...")
    s2_df = pl.read_csv(os.path.join(TRAIN_DIR, "train_source2.tsv"), separator="\t", columns=["entity_id", "country"])
    s2_map = dict(zip(s2_df["entity_id"], s2_df["country"]))
    
    print("Loading S3 countries...")
    s3_df = pl.read_csv(os.path.join(TRAIN_DIR, "train_source3.tsv"), separator="\t", columns=["entity_id", "country"])
    s3_map = dict(zip(s3_df["entity_id"], s3_df["country"]))
    
    gt_df = pl.read_csv(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), separator="\t")
    
    mismatched_country_pairs = 0
    total_checked_pairs = 0
    country_pair_counts = Counter()
    missing_target_ids = 0
    
    for row in gt_df.iter_rows(named=True):
        m_str = row["matched_entity_ids"]
        if not m_str:
            continue
        s1_id = row["source1_entity_id"]
        s1_c = s1_map.get(s1_id)
        
        for tid in m_str.split(","):
            tid = tid.strip()
            total_checked_pairs += 1
            if tid.startswith("S2-"):
                t_c = s2_map.get(tid)
            elif tid.startswith("S3-"):
                t_c = s3_map.get(tid)
            else:
                t_c = None
                
            if t_c is None:
                missing_target_ids += 1
            else:
                country_pair_counts[(s1_c, t_c)] += 1
                if s1_c != t_c:
                    mismatched_country_pairs += 1

    print(f"Total true match pairs checked: {total_checked_pairs:,}")
    print(f"Missing target IDs from S2/S3: {missing_target_ids}")
    print(f"Country consistency in true matches:")
    for (c1, c2), count in country_pair_counts.items():
        print(f"  S1: {c1} -> S2/S3: {c2} = {count:,} pairs ({count / total_checked_pairs * 100:.2f}%)")
    print(f"Total mismatched country pairs: {mismatched_country_pairs:,}")
    
    return {
        "total_true_pairs": total_checked_pairs,
        "mismatched_country_pairs": mismatched_country_pairs,
        "missing_target_ids": missing_target_ids,
        "country_pair_counts": {f"{k[0]}_to_{k[1]}": v for k, v in country_pair_counts.items()}
    }

def main():
    gt_res = audit_ground_truth()
    country_res = audit_countries()
    cross_res = audit_cross_country_matches()
    
    with open("eda_deep_summary.json", "w") as f:
        json.dump({
            "ground_truth": gt_res,
            "countries": country_res,
            "cross_country": cross_res
        }, f, indent=2)
    print("\nSaved eda_deep_summary.json successfully.")

if __name__ == "__main__":
    main()
