"""
Phase 1: Comprehensive Data Audit & EDA Script
Uses polars and streaming Python to accurately analyze large datasets without memory issues.
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

def inspect_file(filepath):
    print(f"\n==========================================")
    print(f"Auditing file: {filepath}")
    file_size_bytes = os.path.getsize(filepath)
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB ({file_size_bytes:,} bytes)")
    
    # Read schema and first few rows
    df_sample = pl.read_csv(filepath, separator="\t", n_rows=5, infer_schema_length=100)
    print(f"Columns: {df_sample.columns}")
    print(f"Schema: {dict(df_sample.schema)}")
    print(f"Sample 3 rows:\n{df_sample.head(3)}")
    
    # Full scan for counts & nulls
    # Using polars scan for memory efficiency
    q = pl.scan_csv(filepath, separator="\t", infer_schema_length=0) # read all as utf8 strings
    
    total_rows = q.select(pl.len()).collect().item()
    print(f"Total rows: {total_rows:,}")
    
    null_exprs = [pl.col(c).is_null().sum().alias(f"{c}_null") for c in df_sample.columns]
    empty_exprs = [(pl.col(c) == "").sum().alias(f"{c}_empty") for c in df_sample.columns]
    
    counts_df = q.select(null_exprs + empty_exprs).collect()
    print("Null/empty counts:")
    for c in df_sample.columns:
        null_c = counts_df[f"{c}_null"][0]
        empty_c = counts_df[f"{c}_empty"][0]
        print(f"  - {c}: {null_c:,} nulls, {empty_c:,} empty strings (total missing: {null_c + empty_c:,})")
        
    # Check ID uniqueness if entity_id or source1_entity_id exists
    id_col = "entity_id" if "entity_id" in df_sample.columns else ("source1_entity_id" if "source1_entity_id" in df_sample.columns else None)
    if id_col:
        unique_ids = q.select(pl.col(id_col).n_unique()).collect().item()
        duplicates = total_rows - unique_ids
        print(f"ID uniqueness for '{id_col}': {unique_ids:,} unique, {duplicates:,} duplicates")
        
        # ID prefix checks
        prefixes = q.select(pl.col(id_col).str.slice(0, 3).alias("prefix")).group_by("prefix").len().collect()
        print(f"ID prefixes:\n{prefixes}")

    # Country distribution if present
    if "country" in df_sample.columns:
        countries = q.group_by("country").len().collect().sort("len", descending=True)
        print(f"Country distribution:\n{countries}")

    return {
        "filepath": filepath,
        "size_mb": file_size_mb,
        "rows": total_rows,
        "columns": df_sample.columns,
        "null_counts": {c: int(counts_df[f"{c}_null"][0] + counts_df[f"{c}_empty"][0]) for c in df_sample.columns}
    }

def main():
    files_to_check = [
        os.path.join(TRAIN_DIR, "train_source1.tsv"),
        os.path.join(TRAIN_DIR, "train_source2.tsv"),
        os.path.join(TRAIN_DIR, "train_source3.tsv"),
        os.path.join(TRAIN_DIR, "train_ground_truth.tsv"),
        os.path.join(TEST_DIR, "test_source1.tsv"),
        os.path.join(TEST_DIR, "test_source2.tsv"),
        os.path.join(TEST_DIR, "test_source3.tsv"),
    ]
    
    summary = {}
    for f in files_to_check:
        if os.path.exists(f):
            summary[os.path.basename(f)] = inspect_file(f)
        else:
            print(f"MISSING: {f}")
            
    with open("eda_summary_basic.json", "w") as out:
        json.dump(summary, out, indent=2)
    print("\nSaved eda_summary_basic.json successfully.")

if __name__ == "__main__":
    main()
