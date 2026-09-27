"""
Phase 2: Comprehensive Normalization Experiments & Collision Analysis Runner.
Evaluates true-match recovery on validation set, collision group sizes,
address components, and multilingual script distributions.
"""

import os
import sys
import json
import time
import unicodedata
from collections import Counter, defaultdict
import polars as pl
import numpy as np

# Ensure parent directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config
from src.normalize import (
    normalize_unicode,
    clean_text,
    strip_legal_suffixes,
    normalize_domain_or_handle,
    to_alphanumeric,
    to_sorted_tokens,
    tokenize,
    extract_numbers,
    extract_postal_code,
    normalize_address,
    normalize_country
)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')


def get_script_name(text):
    """Identifies primary Unicode script of a string."""
    scripts = Counter()
    for c in text:
        if c.isalpha():
            name = unicodedata.name(c, "")
            if "LATIN" in name:
                scripts["Latin"] += 1
            elif "DEVANAGARI" in name:
                scripts["Devanagari"] += 1
            elif "TAMIL" in name:
                scripts["Tamil"] += 1
            elif "TELUGU" in name:
                scripts["Telugu"] += 1
            elif "BENGALI" in name:
                scripts["Bengali"] += 1
            elif "GUJARATI" in name:
                scripts["Gujarati"] += 1
            elif "KANNADA" in name:
                scripts["Kannada"] += 1
            elif "MALAYALAM" in name:
                scripts["Malayalam"] += 1
            elif "ORIYA" in name or "ODIA" in name:
                scripts["Odia"] += 1
            else:
                scripts["Other"] += 1
    if not scripts:
        return "Numeric/Symbolic"
    return scripts.most_common(1)[0][0]


def run_experiments(num_val_pairs=30000, num_collision_entities=100000):
    print("=========================================================")
    print("PHASE 2: NORMALIZATION EXPERIMENTS & COLLISION BENCHMARK")
    print("=========================================================")

    # 1. Load validation ground truth and S1 entities
    print(f"\n1. Loading Validation Split from {config.SPLITS_DIR}...")
    val_s1_df = pl.read_parquet(os.path.join(config.SPLITS_DIR, "val_s1_split.parquet"))
    val_gt_df = pl.read_parquet(os.path.join(config.SPLITS_DIR, "val_ground_truth.parquet"))
    val_s1_set = set(val_s1_df["entity_id"])
    print(f"Validation S1 entities: {len(val_s1_set):,}")

    # Load S1 source text for validation IDs
    print("Loading Source 1 records for validation...")
    s1_full = pl.read_csv(config.TRAIN_SOURCE1_PATH, separator="\t")
    val_s1_records = s1_full.filter(pl.col("entity_id").is_in(list(val_s1_set)))
    s1_dict = {
        r["entity_id"]: (r["business_name"] or "", r["business_address"] or "", r["country"] or "")
        for r in val_s1_records.iter_rows(named=True)
    }

    # Extract target IDs needed from S2 and S3
    needed_s2_ids = set()
    needed_s3_ids = set()
    val_pairs_list = []

    for row in val_gt_df.iter_rows(named=True):
        s1_id = row["source1_entity_id"]
        m_str = row["matched_entity_ids"]
        if not m_str:
            continue
        for tid in m_str.split(","):
            tid = tid.strip()
            if tid.startswith("S2-"):
                needed_s2_ids.add(tid)
            elif tid.startswith("S3-"):
                needed_s3_ids.add(tid)
            val_pairs_list.append((s1_id, tid))
            if len(val_pairs_list) >= num_val_pairs * 2:
                break
        if len(val_pairs_list) >= num_val_pairs * 2:
            break

    print(f"Target pairs needed: {len(val_pairs_list):,}")
    print(f"Loading {len(needed_s2_ids):,} S2 records and {len(needed_s3_ids):,} S3 records...")

    # Load S2 and S3 targets
    s2_full = pl.read_csv(config.TRAIN_SOURCE2_PATH, separator="\t")
    s2_records = s2_full.filter(pl.col("entity_id").is_in(list(needed_s2_ids)))
    target_dict = {
        r["entity_id"]: (r["business_name"] or "", r["business_address"] or "", r["country"] or "")
        for r in s2_records.iter_rows(named=True)
    }

    s3_full = pl.read_csv(config.TRAIN_SOURCE3_PATH, separator="\t")
    s3_records = s3_full.filter(pl.col("entity_id").is_in(list(needed_s3_ids)))
    for r in s3_records.iter_rows(named=True):
        target_dict[r["entity_id"]] = (r["business_name"] or "", r["business_address"] or "", r["country"] or "")

    # Filter to pairs where both records are loaded
    valid_val_pairs = []
    for s1_id, tid in val_pairs_list:
        if s1_id in s1_dict and tid in target_dict:
            valid_val_pairs.append((s1_dict[s1_id], target_dict[tid]))
            if len(valid_val_pairs) >= num_val_pairs:
                break

    N = len(valid_val_pairs)
    print(f"Successfully assembled {N:,} validation true-match pairs for evaluation.")

    # ---------------------------------------------------------
    # 2. Evaluate True-Match Name Recovery Across Representations
    # ---------------------------------------------------------
    print("\n2. Evaluating True-Match Recovery by Representation...")
    rec_raw = 0
    rec_casefold = 0
    rec_clean = 0
    rec_no_suffix = 0
    rec_alpha = 0
    rec_sorted = 0
    rec_domain_handle = 0
    rec_any_exact = 0

    # Address metrics
    addr_exact_clean = 0
    num_token_overlap = 0
    num_both_present = 0
    pin_both_present = 0
    pin_exact_match = 0
    addr_jaccards = []

    # Multilingual script counts on true matches
    scripts_counter = Counter()

    t0 = time.time()
    for (s1_name, s1_addr, s1_c), (t_name, t_addr, t_c) in valid_val_pairs:
        # Script analysis
        script = get_script_name(s1_name)
        scripts_counter[script] += 1
        t_script = get_script_name(t_name)
        if t_script != script:
            scripts_counter[f"{script}_to_{t_script}"] += 1

        # Name representations
        matched_any = False

        # Raw
        if s1_name == t_name:
            rec_raw += 1
            matched_any = True

        # Casefold
        s1_cf = s1_name.casefold()
        t_cf = t_name.casefold()
        if s1_cf == t_cf:
            rec_casefold += 1
            matched_any = True

        # Clean
        s1_cl = clean_text(s1_name)
        t_cl = clean_text(t_name)
        if s1_cl == t_cl and s1_cl != "":
            rec_clean += 1
            matched_any = True

        # Suffix-cleaned
        s1_ns = strip_legal_suffixes(s1_name)
        t_ns = strip_legal_suffixes(t_name)
        if s1_ns == t_ns and s1_ns != "":
            rec_no_suffix += 1
            matched_any = True

        # Alphanumeric
        s1_al = to_alphanumeric(s1_ns or s1_cl)
        t_al = to_alphanumeric(t_ns or t_cl)
        if s1_al == t_al and s1_al != "":
            rec_alpha += 1
            matched_any = True

        # Sorted tokens
        s1_st = to_sorted_tokens(s1_name, remove_suffixes=True)
        t_st = to_sorted_tokens(t_name, remove_suffixes=True)
        if s1_st == t_st and s1_st != "":
            rec_sorted += 1
            matched_any = True

        # Domain / Handle
        s1_dh = normalize_domain_or_handle(s1_name)
        t_dh = normalize_domain_or_handle(t_name)
        if s1_dh == t_dh and s1_dh != "":
            rec_domain_handle += 1
            matched_any = True

        if matched_any:
            rec_any_exact += 1

        # Address analysis
        s1_a_norm = normalize_address(s1_addr)
        t_a_norm = normalize_address(t_addr)

        if not s1_a_norm["is_address_missing"] and not t_a_norm["is_address_missing"]:
            if s1_a_norm["address_clean"] == t_a_norm["address_clean"]:
                addr_exact_clean += 1

            s1_nums = set(s1_a_norm["address_numbers"])
            t_nums = set(t_a_norm["address_numbers"])
            if s1_nums and t_nums:
                num_both_present += 1
                if s1_nums & t_nums:
                    num_token_overlap += 1

            p1 = s1_a_norm["postal_code"]
            p2 = t_a_norm["postal_code"]
            if p1 and p2:
                pin_both_present += 1
                if p1 == p2:
                    pin_exact_match += 1

            s1_toks = set(s1_a_norm["address_tokens"])
            t_toks = set(t_a_norm["address_tokens"])
            if s1_toks or t_toks:
                jacc = len(s1_toks & t_toks) / len(s1_toks | t_toks)
                addr_jaccards.append(jacc)

    elapsed_eval = time.time() - t0
    throughput = (N * 2) / elapsed_eval
    print(f"Evaluated {N:,} pairs in {elapsed_eval:.2f}s ({throughput:,.0f} records/sec)")

    print("\n--- TRUE-MATCH RECOVERY RESULTS (N = {:,}) ---".format(N))
    print(f"1. Raw Exact Match:              {rec_raw:,} ({rec_raw / N * 100:.2f}%)")
    print(f"2. Casefold Exact Match:         {rec_casefold:,} ({rec_casefold / N * 100:.2f}%)")
    print(f"3. Clean Exact Match:            {rec_clean:,} ({rec_clean / N * 100:.2f}%)")
    print(f"4. Suffix-Stripped Exact Match:  {rec_no_suffix:,} ({rec_no_suffix / N * 100:.2f}%)")
    print(f"5. Alphanumeric Exact Match:     {rec_alpha:,} ({rec_alpha / N * 100:.2f}%)")
    print(f"6. Sorted-Tokens Exact Match:    {rec_sorted:,} ({rec_sorted / N * 100:.2f}%)")
    print(f"7. Domain/Handle Exact Match:    {rec_domain_handle:,} ({rec_domain_handle / N * 100:.2f}%)")
    print(f"8. ANY Exact Match (Union):      {rec_any_exact:,} ({rec_any_exact / N * 100:.2f}%)")

    print("\n--- ADDRESS COMPONENT RESULTS ---")
    print(f"Exact Normalized Address Match:   {addr_exact_clean:,} / {N:,} ({addr_exact_clean / N * 100:.2f}%)")
    if num_both_present > 0:
        print(f"House/Plot Number Match (when present in both): {num_token_overlap:,} / {num_both_present:,} ({num_token_overlap / num_both_present * 100:.2f}%)")
    if pin_both_present > 0:
        print(f"Postal Code Match (when present in both):       {pin_exact_match:,} / {pin_both_present:,} ({pin_exact_match / pin_both_present * 100:.2f}%)")
    if addr_jaccards:
        print(f"Mean Address Token Jaccard:       {np.mean(addr_jaccards):.4f} (Median: {np.median(addr_jaccards):.4f})")

    # ---------------------------------------------------------
    # 3. Collision Analysis on 100,000 Source 1 Entities
    # ---------------------------------------------------------
    print(f"\n3. Performing Collision Analysis on {num_collision_entities:,} Source 1 Entities...")
    sample_s1 = val_s1_records.slice(0, num_collision_entities)
    
    rep_names = {
        "raw": [r["business_name"] or "" for r in sample_s1.iter_rows(named=True)],
        "clean": [clean_text(r["business_name"]) for r in sample_s1.iter_rows(named=True)],
        "suffix_stripped": [strip_legal_suffixes(r["business_name"]) for r in sample_s1.iter_rows(named=True)],
        "alphanumeric": [to_alphanumeric(strip_legal_suffixes(r["business_name"])) for r in sample_s1.iter_rows(named=True)],
        "sorted_tokens": [to_sorted_tokens(r["business_name"], remove_suffixes=True) for r in sample_s1.iter_rows(named=True)]
    }

    collision_stats = {}
    print(f"{'Representation':<20} | {'Unique Values':<14} | {'Collision Rate':<15} | {'Max Group':<10} | {'Top Colliding String'}")
    print("-" * 90)

    for rep_name, values in rep_names.items():
        val_counts = Counter(values)
        num_unique = len(val_counts)
        total_items = len(values)
        collision_rate = 1.0 - (num_unique / total_items)
        
        # Most common
        top_str, max_group = val_counts.most_common(1)[0] if val_counts else ("", 0)
        
        # Group size percentiles
        sizes = list(val_counts.values())
        p50 = float(np.median(sizes))
        p95 = float(np.percentile(sizes, 95))
        p99 = float(np.percentile(sizes, 99))
        
        collision_stats[rep_name] = {
            "total_entities": total_items,
            "unique_values": num_unique,
            "collision_rate_pct": round(collision_rate * 100, 2),
            "max_group_size": max_group,
            "top_string": top_str[:30],
            "median_group_size": p50,
            "p95_group_size": p95,
            "p99_group_size": p99
        }
        
        print(f"{rep_name:<20} | {num_unique:<14,} | {collision_rate * 100:<14.2f}% | {max_group:<10,} | '{top_str[:30]}'")

    # ---------------------------------------------------------
    # 4. Multilingual Script Distribution Summary
    # ---------------------------------------------------------
    print("\n--- MULTILINGUAL SCRIPT DISTRIBUTION ON TRUE MATCHES ---")
    for s_name, count in scripts_counter.most_common(10):
        print(f"  {s_name:<25}: {count:,} ({count / N * 100:.2f}%)")

    results = {
        "validation_pairs_evaluated": N,
        "throughput_records_per_sec": round(throughput, 1),
        "recovery_rates_pct": {
            "raw": round(rec_raw / N * 100, 2),
            "casefold": round(rec_casefold / N * 100, 2),
            "clean": round(rec_clean / N * 100, 2),
            "suffix_stripped": round(rec_no_suffix / N * 100, 2),
            "alphanumeric": round(rec_alpha / N * 100, 2),
            "sorted_tokens": round(rec_sorted / N * 100, 2),
            "domain_handle": round(rec_domain_handle / N * 100, 2),
            "any_exact_union": round(rec_any_exact / N * 100, 2)
        },
        "address_metrics": {
            "exact_clean_pct": round(addr_exact_clean / N * 100, 2),
            "house_number_match_pct": round(num_token_overlap / num_both_present * 100, 2) if num_both_present else 0.0,
            "postal_code_match_pct": round(pin_exact_match / pin_both_present * 100, 2) if pin_both_present else 0.0,
            "mean_jaccard": round(float(np.mean(addr_jaccards)), 4) if addr_jaccards else 0.0,
            "median_jaccard": round(float(np.median(addr_jaccards)), 4) if addr_jaccards else 0.0
        },
        "collision_analysis": collision_stats,
        "script_distribution": dict(scripts_counter.most_common(10))
    }

    out_path = os.path.join(config.PROJECT_ROOT, "phase2_experiment_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved phase2_experiment_results.json successfully.")
    return results


if __name__ == "__main__":
    run_experiments()
