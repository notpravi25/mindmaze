# Phase 2: Validation Framework & Text Normalization Engine Report
**Amazon ML Challenge 2026 — Business Entity Resolution**  
**Evaluation Date:** September 27, 2026  
**Status:** Completed & Tested

---

## 1. Entity-Level Validation Split Analysis

A deterministic, zero-leakage, stratified validation split was created at the **Source 1 entity level** using `src/split.py`.

### 1.1 Split Parameters
- **Validation Fraction:** `0.20` (20.0%)
- **Random Seed:** `42`
- **Stratification Key:** `(country, match_cardinality_bin)`
- **Artifacts Saved:**
  - `data/splits/train_s1_split.parquet` (1,765,456 rows)
  - `data/splits/val_s1_split.parquet` (441,365 rows)
  - `data/splits/val_ground_truth.parquet` (441,365 rows)
  - `data/splits/split_metadata.json`

### 1.2 Distribution Comparison (Train vs. Validation)

| Metric | Full Training Set | Train Split (80%) | Validation Split (20%) | Conformance |
|---|---|---|---|---|
| **Total S1 Entities** | 2,206,821 | **1,765,456** | **441,365** | Exact 80.00% / 20.00% |
| **Singletons (0 matches)** | 123,247 (5.585%) | 98,598 (**5.585%**) | 24,649 (**5.585%**) | **Exact Match to 3 decimals** |
| **Single Match (1 match)** | 119,157 (5.399%) | 95,325 (**5.399%**) | 23,832 (**5.400%**) | **Exact Match** |
| **Multi-Match ($\ge 2$ matches)** | 1,964,417 (89.016%) | 1,571,533 (**89.016%**) | 392,884 (**89.016%**) | **Exact Match** |
| **US Entity Proportion** | 59.98% | **59.98%** | **59.98%** | **Exact Match** |
| **India Entity Proportion** | 40.02% | **40.02%** | **40.02%** | **Exact Match** |
| **Mean Matches per S1** | 3.4613 | **3.4614** | **3.4607** | **Identical ($\Delta < 0.001$)** |
| **Total True Match Pairs** | 7,638,365 | 6,110,933 | 1,527,432 | Total preserved |

### 1.3 Zero-Leakage Verification
- **S1 Overlap:** `len(set(train_s1_ids) & set(val_s1_ids)) == 0` (Confirmed: **0 overlapping IDs**).
- **Relational Integrity:** All match target relationships for every validation S1 are completely isolated inside `val_ground_truth.parquet`. No entity has matches split across train and validation.

---

## 2. Text Normalization Representations Implemented

The normalization engine was implemented in [`src/normalize.py`](file:///c:/Users/user/Desktop/mindmaze/src/normalize.py). It maintains the original fields intact while computing specialized, deterministic views:

### 2.1 Business Name Representations
1. `name_original`: Raw, unadulterated string.
2. `name_nfkd`: Unicode NFKD decomposed string.
3. `name_casefold`: Multilingual `casefold()` with selective Latin diacritic stripping (accents like `é` $\rightarrow$ `e` are folded while Indic vowels and matras are strictly preserved).
4. `name_clean`: Punctuation normalized (e.g. `&` $\rightarrow$ `and`, noise brackets `[[`, `]]`, `##`, `<<` stripped, non-alphanumeric separators turned to single spaces, whitespace collapsed).
5. `name_no_legal_suffix`: Configurable legal suffix removal on word/token boundaries at end or within name (covering US: `inc`, `llc`, `corp`; India: `pvt ltd`, `ltd`; France: `sarl`, `sas`, `sasu`, `sci`, `sa`; and general corporate descriptors: `enterprises`, `solutions`, `technologies`, `services`, `group`).
6. `name_domain`: Domain extension stripper (`.com`, `.org`, `.net`, `.in`, `.co.in`, `.fr`, etc.) and social handle stripper (`@smartraj` $\rightarrow$ `smartraj`).
7. `name_alphanumeric`: Compact alphanumeric representation (`ABC-Pvt. Ltd.` $\rightarrow$ `abcpvtltd` or `abc`).
8. `name_tokens`: Tokenized list of words preserving short tokens (e.g. `A`, `AI`, `UK`, `3D`).
9. `name_sorted_tokens`: Word-order invariant sorted token string (`XX Apex Nippon` and `XX Nippon Apex` $\rightarrow$ `apex nippon xx`).

### 2.2 Address Representations & Extracted Components
1. `address_original`: Raw string intact.
2. `address_clean`: Lowercased/casefolded, accents normalized, noise stripped, abbreviations standardized (`st` $\rightarrow$ `street`, `rd` $\rightarrow$ `road`, `ave` $\rightarrow$ `avenue`, `dr` $\rightarrow$ `drive`, `blvd` $\rightarrow$ `boulevard`, `r.` $\rightarrow$ `rue`, `bd` $\rightarrow$ `boulevard`).
3. `address_alphanumeric`: Compact alphanumeric-only representation.
4. `address_tokens`: Tokenized list of standardized address tokens.
5. `address_numbers`: Reusable numerical identifier extractor (house, plot, building, floor, street number, flat number; normalizes ordinals like `2nd` $\rightarrow$ `2`).
6. `postal_code`: Generic postal code extractor (5 to 6 digits, e.g. 5-digit US ZIP, 6-digit India PIN, 5-digit France postal code).
7. `is_address_missing`: Boolean indicator flag when address is None or empty.

### 2.3 Country Representations
1. `country_original`: Raw string.
2. `country_normalized`: Stripped, uppercase, dynamic string (no fixed whitelist; works identically on US, India, France, and any future country).

---

## 3. Name Normalization Experiments (True-Match Recovery)

We evaluated the exact match recovery rate of true matches on a held-out sample of **30,000 validation true-match pairs** in [`run_phase2_experiments.py`](file:///c:/Users/user/Desktop/mindmaze/run_phase2_experiments.py):

| Representation | True Matches Recovered | Recovery Rate (%) | Gain over Raw | Key Takeaway |
|---|---|---|---|---|
| **Raw Exact Match** | 1,306 | **4.35%** | Baseline | 95.6% of true matches are lost by raw comparison. |
| **Casefold Exact Match** | 3,121 | **10.40%** | $+6.05\%$ | Casing alone recovers few matches. |
| **Clean Exact Match** | 7,804 | **26.01%** | $+21.66\%$ | Stripping punctuation, ampersands, noise brackets jumps recovery $6\times$. |
| **Suffix-Stripped Exact Match** | 14,405 | **48.02%** | $+43.67\%$ | **Removing legal suffixes jumps recovery $11\times$!** |
| **Alphanumeric Exact Match** | 14,491 | **48.30%** | $+43.95\%$ | Bridges minor punctuation differences. |
| **Sorted-Tokens Exact Match** | 15,171 | **50.57%** | $+46.22\%$ | **Word-order invariance recovers over half of all true matches directly!** |
| **Domain/Handle Exact Match** | 7,809 | **26.03%** | $+21.68\%$ | Recovers URLs and social handles. |
| **ANY Exact Match (Union)** | **15,262** | **50.87%** | **$+46.52\%$** | **50.87% of all true matches are recovered by exact normalization alone.** |

---

## 4. Collision Analysis (Checking for Over-Normalization)

We evaluated collision behavior across **100,000 Source 1 entities** to confirm that aggressive normalization does not collapse distinct businesses:

| Representation | Unique Values | Collision Rate | Max Collision Group | 95th Percentile | Top Colliding String |
|---|---|---|---|---|---|
| **raw** | 90,365 | **9.64%** | 16 | 2.0 | `'Eye Group'` |
| **clean** | 89,761 | **10.24%** | 16 | 2.0 | `'eye group'` |
| **suffix_stripped** | 82,750 | **17.25%** | 31 | 2.0 | `'meridian'` |
| **alphanumeric** | 82,740 | **17.26%** | 31 | 2.0 | `'meridian'` |
| **sorted_tokens** | 82,697 | **17.30%** | 31 | 2.0 | `'meridian'` |

### Collision Finding:
- Suffix stripping and token sorting increase the collision rate modestly from **9.64% to 17.30%**.
- The maximum collision group size is only **31 entities** (out of 100,000), and 95% of groups have $\le 2$ entities.
- **Over-normalization collapse does not occur.** The representation remains highly discriminative while doubling true-match recovery.

---

## 5. Address Normalization & Component Analysis

Evaluated across the 30,000 validation true-match pairs:

| Address Metric | Measured Value | Insight |
|---|---|---|
| **Exact Normalized Address Match** | **11.55%** (3,466 pairs) | Even after normalization, full address equality is rare due to formatting. |
| **House / Building Number Match** | **87.52%** (22,775 / 26,023) | When numbers exist in both addresses, **87.5% share identical numbers!** |
| **Postal Code Match (when present in both)** | **88.79%** (1,291 / 1,454) | Very high agreement rate; highly reliable verification feature. |
| **Mean Address Token Jaccard** | **0.6890** (Median: **0.7000**) | Address tokens share strong overlap across true matches. |

---

## 6. Multilingual & Script Analysis

Analysis of scripts across true matches:
- **Source 1 Language/Script:** 100% Latin (English strings).
- **Target Cross-Script Matches in S2 and S3:**
  - Latin $\rightarrow$ Devanagari (Hindi): **1,177 pairs (3.92%)**
  - Latin $\rightarrow$ Telugu: **212 pairs (0.71%)**
  - Latin $\rightarrow$ Bengali: **166 pairs (0.55%)**
  - Latin $\rightarrow$ Kannada: **152 pairs (0.51%)**
  - Latin $\rightarrow$ Tamil: **151 pairs (0.50%)**
  - Latin $\rightarrow$ Gujarati: **130 pairs (0.43%)**
  - Latin $\rightarrow$ Malayalam: **110 pairs (0.37%)**
  - Latin $\rightarrow$ Odia: **29 pairs (0.10%)**
  - **Total Cross-Script Matches:** **~7.2%** of all true matches!

### Normalization Behavior on Non-Latin Scripts:
- `normalize_unicode`: Selectively folds Latin accents (`Président` $\rightarrow$ `President`) while preserving Indic matras and vowel signs intact.
- **The Address Bridge:** In all cross-script match pairs, the business name is transliterated into regional Indic script, but **the address is written in Latin English with matching plot/building numbers and street names**. Address-based blocking and numerical features are essential to recover this 7.2% recall ceiling.

---

## 7. Performance & Throughput Benchmarks

- **Pure Python Normalization & Comparison:** Evaluated 30,000 pairs (60,000 records normalized and compared) in **5.88 seconds** $\rightarrow$ **10,202 records/sec**.
- **Memory Overhead:** Negligible; streaming and vectorized representations prevent in-memory duplication.
- **Full Scale Extrapolation:** Normalizing 1.73M test records will take under 3 minutes using chunked processing.

---

## 8. Unit Tests

Unit tests implemented in [`tests/test_normalize.py`](file:///c:/Users/user/Desktop/mindmaze/tests/test_normalize.py) verify:
- Legal suffix removal across US, UK, India, and France.
- Boundary-awareness (words like `salvage` and `incorporate` are undamaged).
- Punctuation and noise bracket stripping (`[[LLC]]`, `##2476`, `<<`).
- Domain name extraction (`strategicpraetorian.com` $\rightarrow$ `strategicpraetorian`).
- Social handles (`@smartraj` $\rightarrow$ `smartraj`, `@apex_labs` $\rightarrow$ `apex labs`).
- Word-order invariance (`XX Apex Nippon` $\equiv$ `XX Nippon Apex`).
- Unicode accent folding on French with Indic vowel preservation on Hindi/Tamil.
- Address number and postal code extraction.
- Deterministic behavior across runs.

**Result:** `Ran 12 tests in 0.001s: OK (12 passed, 0 failed)`.

---

## 9. Recommended Representations for Phase 3 (Blocking & Features)

Based on actual measured validation evidence:

### For Candidate Blocking (Phase 3/4):
1. **Blocking Block 1 (Exact Normalized Sorted Tokens):**  
   `country + name_sorted_tokens`  
   *Why:* Recovers **50.57%** of true matches immediately with a low collision rate (17.30%, max group size 31).
2. **Blocking Block 2 (Rare Name Tokens):**  
   `country + rare_name_token` (using tokens with corpus frequency $< 10^{-4}$, excluding stopwords).  
   *Why:* Recovers names with minor typos, legal suffix variations, and additional descriptors.
3. **Blocking Block 3 (Domain / Handle Core):**  
   `country + name_domain`  
   *Why:* Recovers domain-name and handle variants (e.g. `strategicpraetorian.com`).
4. **Blocking Block 4 (Address Number + Street / City Token):**  
   `country + address_number + address_token`  
   *Why:* Recovers the **7.2%** of true matches where the business name is transliterated into Devanagari/Tamil/Telugu but the address contains identical house/plot numbers.

### For Pairwise Matching Features:
1. `name_clean_jaccard` (token overlap on clean names)
2. `name_sorted_tokens_exact` (binary match flag)
3. `name_char_3gram_similarity` (character trigram cosine)
4. `name_length_diff_ratio`
5. `address_clean_jaccard`
6. `house_number_exact_match` (binary flag; 87.5% precision indicator)
7. `postal_code_match` (1 = match, 0 = mismatch, -1 = missing)
8. `is_address_missing` (indicator flag)
9. `source_indicator` (`is_s2` vs `is_s3`)

---
*End of Phase 2 Report. Awaiting user review before Phase 3.*
