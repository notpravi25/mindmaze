"""
Validation split module: Creates and manages a reproducible, entity-level
train/validation split without data leakage.
"""

import os
import sys
import json
import polars as pl
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# Ensure parent directory is in path when run standalone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')


def create_entity_validation_split(
    val_fraction=config.VALIDATION_FRACTION,
    random_seed=config.RANDOM_SEED,
    output_dir=config.SPLITS_DIR
):
    """
    Creates a deterministic, entity-level validation split from Source 1 entities.
    Stratifies across (country, match_cardinality_bin) to guarantee matched distributions.
    """
    print(f"Creating Entity-Level Validation Split...")
    print(f"  Validation Fraction: {val_fraction:.2f}")
    print(f"  Random Seed: {random_seed}")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load S1 entities and country
    print("Loading Source 1 entities and ground truth...")
    s1_df = pl.read_csv(
        config.TRAIN_SOURCE1_PATH,
        separator="\t",
        columns=["entity_id", "country"]
    )
    
    gt_df = pl.read_csv(
        config.TRAIN_GROUND_TRUTH_PATH,
        separator="\t",
        columns=["source1_entity_id", "matched_entity_ids"]
    )

    # 2. Join country with ground truth to determine match count & strat key
    # Compute match count
    # Count comma separated items in matched_entity_ids
    merged = s1_df.join(gt_df, left_on="entity_id", right_on="source1_entity_id", how="left")
    
    # Calculate match count per S1
    match_counts = []
    strat_keys = []
    
    for row in merged.iter_rows(named=True):
        m_str = row["matched_entity_ids"]
        country = row["country"]
        
        if m_str is None or m_str.strip() == "":
            cnt = 0
            bin_label = "0_singleton"
        else:
            ids = [x for x in m_str.split(",") if x.strip()]
            cnt = len(ids)
            if cnt == 1:
                bin_label = "1_single"
            elif cnt in (2, 3):
                bin_label = "2_3_multi"
            elif cnt in (4, 5):
                bin_label = "4_5_multi"
            else:
                bin_label = "6plus_multi"
                
        match_counts.append(cnt)
        strat_keys.append(f"{country}_{bin_label}")

    merged = merged.with_columns([
        pl.Series("match_count", match_counts),
        pl.Series("strat_key", strat_keys)
    ])

    total_s1 = merged.height
    print(f"Total Source 1 entities loaded: {total_s1:,}")

    # 3. Stratified Split on S1 entities
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=random_seed)
    indices = np.arange(total_s1)
    train_idx, val_idx = next(splitter.split(indices, strat_keys))

    train_merged = merged[train_idx]
    val_merged = merged[val_idx]

    train_ids = set(train_merged["entity_id"])
    val_ids = set(val_merged["entity_id"])

    # 4. Verify ZERO LEAKAGE
    overlap = train_ids.intersection(val_ids)
    assert len(overlap) == 0, f"FATAL: Overlap detected between train and val S1 IDs! Count: {len(overlap)}"
    assert len(train_ids) + len(val_ids) == total_s1, "FATAL: Total IDs mismatch after split!"
    print(f"Zero-leakage check PASSED. Overlap = {len(overlap)}")

    # 5. Compute Detailed Split Statistics
    def compute_stats(df, label):
        n = df.height
        c_dist = {r[0]: int(r[1]) for r in df.group_by("country").len().iter_rows()}
        singletons = df.filter(pl.col("match_count") == 0).height
        single_m = df.filter(pl.col("match_count") == 1).height
        multi_m = df.filter(pl.col("match_count") >= 2).height
        
        freq = {str(r[0]): int(r[1]) for r in df.group_by("match_count").len().sort("match_count").iter_rows()}
        total_true_pairs = df.select(pl.col("match_count").sum()).item()
        
        return {
            "split": label,
            "total_s1": n,
            "pct_of_total": round(n / total_s1 * 100, 2),
            "country_distribution": c_dist,
            "country_pct": {k: round(v / n * 100, 2) for k, v in c_dist.items()},
            "singletons": singletons,
            "singleton_pct": round(singletons / n * 100, 3),
            "single_matches": single_m,
            "single_match_pct": round(single_m / n * 100, 3),
            "multi_matches": multi_m,
            "multi_match_pct": round(multi_m / n * 100, 3),
            "total_true_pairs": int(total_true_pairs),
            "mean_matches_per_s1": round(float(total_true_pairs / n), 4),
            "match_count_freq": freq
        }

    train_stats = compute_stats(train_merged, "train")
    val_stats = compute_stats(val_merged, "validation")

    print("\n--- SPLIT STATISTICS COMPARISON ---")
    print(f"{'Metric':<28} | {'Train':<15} | {'Validation':<15}")
    print("-" * 64)
    print(f"{'Total S1 Count':<28} | {train_stats['total_s1']:<15,} | {val_stats['total_s1']:<15,}")
    print(f"{'Singletons (0 matches)':<28} | {train_stats['singletons']:<10,} ({train_stats['singleton_pct']}%) | {val_stats['singletons']:<10,} ({val_stats['singleton_pct']}%)")
    print(f"{'Single Match (1 match)':<28} | {train_stats['single_matches']:<10,} ({train_stats['single_match_pct']}%) | {val_stats['single_matches']:<10,} ({val_stats['single_match_pct']}%)")
    print(f"{'Multi-Match (>=2 matches)':<28} | {train_stats['multi_matches']:<10,} ({train_stats['multi_match_pct']}%) | {val_stats['multi_matches']:<10,} ({val_stats['multi_match_pct']}%)")
    print(f"{'US Ratio':<28} | {train_stats['country_pct'].get('US', 0):<15.2f}% | {val_stats['country_pct'].get('US', 0):<15.2f}%")
    print(f"{'India Ratio':<28} | {train_stats['country_pct'].get('India', 0):<15.2f}% | {val_stats['country_pct'].get('India', 0):<15.2f}%")
    print(f"{'Mean Matches / S1':<28} | {train_stats['mean_matches_per_s1']:<15.4f} | {val_stats['mean_matches_per_s1']:<15.4f}")
    print(f"{'Total True Match Pairs':<28} | {train_stats['total_true_pairs']:<15,} | {val_stats['total_true_pairs']:<15,}")

    # 6. Save split files
    print("\nSaving split files to disk...")
    # Save S1 ID lists
    train_ids_df = train_merged.select(["entity_id", "country", "match_count"])
    val_ids_df = val_merged.select(["entity_id", "country", "match_count"])
    
    train_ids_df.write_parquet(os.path.join(output_dir, "train_s1_split.parquet"))
    val_ids_df.write_parquet(os.path.join(output_dir, "val_s1_split.parquet"))

    # Also save separate ground truth for validation
    val_gt = gt_df.filter(pl.col("source1_entity_id").is_in(list(val_ids)))
    val_gt.write_parquet(os.path.join(output_dir, "val_ground_truth.parquet"))
    
    split_meta = {
        "random_seed": random_seed,
        "val_fraction": val_fraction,
        "train": train_stats,
        "validation": val_stats,
        "overlap_count": len(overlap)
    }

    with open(os.path.join(output_dir, "split_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2)

    print(f"Validation split created successfully in {output_dir}")
    return split_meta


def load_validation_ids(output_dir=config.SPLITS_DIR):
    """Loads validation S1 entity IDs as a set for zero-leakage evaluation."""
    val_path = os.path.join(output_dir, "val_s1_split.parquet")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Split file not found at {val_path}. Run create_entity_validation_split first.")
    df = pl.read_parquet(val_path)
    return set(df["entity_id"])


if __name__ == "__main__":
    create_entity_validation_split()
