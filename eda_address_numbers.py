"""
Phase 1: Address and Number Pattern Audit
Measures frequency of house/plot/street numbers, city/state indicators, and address token statistics.
"""

import os
import sys
import json
import re
import polars as pl
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

NUMBER_RE = re.compile(r'\b\d+[a-zA-Z]?\b')

def main():
    print("\n--- AUDITING ADDRESS NUMBER AND TOKEN STATISTICS ---")
    s1_df = pl.read_csv(os.path.join(TRAIN_DIR, "train_source1.tsv"), separator="\t", n_rows=100000)
    
    name_lengths = []
    name_tokens_count = []
    addr_lengths = []
    addr_tokens_count = []
    has_number_count = 0
    
    for row in s1_df.iter_rows(named=True):
        name = row["business_name"] or ""
        addr = row["business_address"] or ""
        
        name_lengths.append(len(name))
        addr_lengths.append(len(addr))
        
        n_toks = len(name.split())
        a_toks = len(addr.split())
        
        name_tokens_count.append(n_toks)
        addr_tokens_count.append(a_toks)
        
        nums = NUMBER_RE.findall(addr)
        if nums:
            has_number_count += 1
            
    total = len(s1_df)
    print(f"Total analyzed S1 records: {total:,}")
    print(f"Name character length: Mean = {np.mean(name_lengths):.1f}, Median = {np.median(name_lengths):.0f}, 95th = {np.percentile(name_lengths, 95):.0f}, Max = {np.max(name_lengths)}")
    print(f"Name word count: Mean = {np.mean(name_tokens_count):.1f}, Median = {np.median(name_tokens_count):.0f}, 95th = {np.percentile(name_tokens_count, 95):.0f}, Max = {np.max(name_tokens_count)}")
    print(f"Address character length: Mean = {np.mean(addr_lengths):.1f}, Median = {np.median(addr_lengths):.0f}, 95th = {np.percentile(addr_lengths, 95):.0f}, Max = {np.max(addr_lengths)}")
    print(f"Address word count: Mean = {np.mean(addr_tokens_count):.1f}, Median = {np.median(addr_tokens_count):.0f}, 95th = {np.percentile(addr_tokens_count, 95):.0f}, Max = {np.max(addr_tokens_count)}")
    print(f"Addresses containing numerical identifiers (house/plot/street/pincode): {has_number_count:,} ({has_number_count / total * 100:.2f}%)")

if __name__ == "__main__":
    main()
