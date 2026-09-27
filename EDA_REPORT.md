# Phase 1: Comprehensive Data Audit & Exploratory Data Analysis (EDA) Report
**Amazon ML Challenge 2026 — Business Entity Resolution**  
**Dataset Evaluation Date:** September 27, 2026  
**Audited Directory:** `dataset/` (containing `dataset/train/` and `dataset/test/`)

---

## 1. Directory Structure & File Verification

All expected files from the official challenge specification were located, verified, and checked for file integrity:

| File Path | File Size | Row Count | Primary Key / ID Column | Schema (Columns) |
|---|---|---|---|---|
| `dataset/train/train_source1.tsv` | 200.34 MB | 2,206,821 | `entity_id` (Prefix `S1-`) | `entity_id`, `business_name`, `business_address`, `country` |
| `dataset/train/train_source2.tsv` | 466.63 MB | 5,034,616 | `entity_id` (Prefix `S2-`) | `entity_id`, `business_name`, `business_address`, `country` |
| `dataset/train/train_source3.tsv` | 480.37 MB | 5,285,603 | `entity_id` (Prefix `S3-`) | `entity_id`, `business_name`, `business_address`, `country` |
| `dataset/train/train_ground_truth.tsv`| 121.13 MB | 2,206,821 | `source1_entity_id` (Prefix `S1-`) | `source1_entity_id`, `matched_entity_ids` |
| `dataset/test/test_source1.tsv` | 166.91 MB | 1,732,544 | `entity_id` (Prefix `S1-`) | `entity_id`, `business_name`, `business_address`, `country` |
| `dataset/test/test_source2.tsv` | 485.86 MB | 4,887,273 | `entity_id` (Prefix `S2-`) | `entity_id`, `business_name`, `business_address`, `country` |
| `dataset/test/test_source3.tsv` | 482.56 MB | 5,082,316 | `entity_id` (Prefix `S3-`) | `entity_id`, `business_name`, `business_address`, `country` |

### Scale Observations
- **Total Training Records:** 12,527,040 (~12.53 Million records across Sources 1, 2, and 3).
- **Total Test Records:** 11,702,133 (~11.70 Million records across Sources 1, 2, and 3).
- **Cartesian Test Space:** $1,732,544 \times (4,887,273 + 5,082,316) \approx \mathbf{17.27 \times 10^{12}}$ (17.27 Trillion candidate pairs).
- **Memory & Scaling Imperative:** Full Cartesian joins or dense pairwise matrices are physically impossible. Multi-stage inverted-index blocking with sparse representation is strictly mandatory.

---

## 2. Integrity Checks & Missing Value Analysis

### 2.1 ID Integrity and Prefixes
- **Duplicate IDs:** Exactly **0** duplicate IDs detected across all 7 dataset files. Every file has 100% unique primary keys.
- **Prefix Conformance:** 
  - Source 1 strictly contains prefix `S1-` (100.0%)
  - Source 2 strictly contains prefix `S2-` (100.0%)
  - Source 3 strictly contains prefix `S3-` (100.0%)
- **Target ID Integrity in Ground Truth:** All target IDs appearing in `train_ground_truth.tsv` were cross-checked against `train_source2.tsv` and `train_source3.tsv`. **0** dangling or missing target IDs were found.
- **No Self-Matches:** Ground truth contains zero self-matches to Source 1.
- **No Duplicate IDs per Match List:** Ground truth contains zero duplicate IDs inside any `matched_entity_ids` list.

### 2.2 Missing / Null Value Breakdown

| File | `entity_id` Missing | `business_name` Missing | `business_address` Missing | `country` Missing |
|---|---|---|---|---|
| `train_source1.tsv` | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| `train_source2.tsv` | 0 (0.00%) | 0 (0.00%) | 168,967 (3.36%) | 0 (0.00%) |
| `train_source3.tsv` | 0 (0.00%) | 0 (0.00%) | 175,916 (3.33%) | 0 (0.00%) |
| `test_source1.tsv`  | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| `test_source2.tsv`  | 0 (0.00%) | 0 (0.00%) | 129,408 (2.65%) | 0 (0.00%) |
| `test_source3.tsv`  | 0 (0.00%) | 0 (0.00%) | 136,098 (2.68%) | 0 (0.00%) |

**Key Takeaway:**
- `business_name` and `country` are 100% complete across all sources in both train and test sets.
- `business_address` has ~2.6% to 3.4% missing values in Sources 2 and 3 (represented as TSV nulls). Source 1 has zero missing addresses.
- Candidate generation and feature engineering must handle missing addresses gracefully (e.g. name-based fallback paths and `is_address_missing` indicator flags).

---

## 3. Country Analysis & The "Zero Cross-Country" Invariant

### 3.1 Country Distribution Comparison (Train vs. Test)

#### Training Set Distribution:
- **US:**
  - `train_source1`: 1,323,633 (59.98%)
  - `train_source2`: 3,016,817 (59.92%)
  - `train_source3`: 3,170,056 (59.98%)
- **India:**
  - `train_source1`: 883,188 (40.02%)
  - `train_source2`: 2,017,799 (40.08%)
  - `train_source3`: 2,115,547 (40.02%)
- **France:** 0 (0.00%) — France is completely absent from the training set.

#### Test Set Distribution:
- **India:**
  - `test_source1`: 809,986 (46.75%)
  - `test_source2`: 2,312,565 (47.32%)
  - `test_source3`: 2,405,000 (47.32%)
- **US:**
  - `test_source1`: 663,106 (38.27%)
  - `test_source2`: 1,871,330 (38.29%)
  - `test_source3`: 1,945,701 (38.28%)
- **France:**
  - `test_source1`: 259,452 (14.98%)
  - `test_source2`: 703,378 (14.39%)
  - `test_source3`: 731,615 (14.40%)

### 3.2 The Zero Cross-Country Invariant
We evaluated all **7,638,365** true match pairs in `train_ground_truth.tsv` against the country of origin for both entities:
- **S1 US $\rightarrow$ S2/S3 US:** 4,578,522 pairs (59.94%)
- **S1 India $\rightarrow$ S2/S3 India:** 3,059,843 pairs (40.06%)
- **Mismatched Country Pairs (Cross-Country):** **0 pairs (0.000%)**

> **CRITICAL ARCHITECTURAL INSIGHT:**  
> Country consistency in ground truth is exactly 100.00%. A business entity in country $A$ never matches a record in country $B$.  
> **Partitioning blocking strictly by exact country match is 100% loss-free.**  
> Furthermore, because France is completely unseen in training, all normalization and blocking logic must be country-generic and rule-agnostic to unseen country names.

---

## 4. Ground Truth Match & Singleton Distribution Analysis

### 4.1 Match Cardinality Breakdown
Across all 2,206,821 Source 1 entities in the training set:

| Match Count per S1 | Number of S1 Entities | Percentage of S1 | Category |
|---|---|---|---|
| **0 matches** | 123,247 | **5.585%** | **Singletons (Unmatched)** |
| **1 match** | 119,157 | **5.399%** | Single Match |
| **2 matches** | 375,212 | **17.002%** | Multi-Match |
| **3 matches** | 530,841 | **24.055%** | Multi-Match |
| **4 matches** | 484,115 | **21.937%** | Multi-Match |
| **5 matches** | 321,957 | **14.589%** | Multi-Match |
| **6 matches** | 164,868 | **7.471%** | Multi-Match |
| **7 matches** | 63,968 | **2.899%** | Multi-Match |
| **8 matches** | 18,680 | **0.846%** | Multi-Match |
| **9 matches** | 4,205 | **0.191%** | Multi-Match |
| **10 matches** | 534 | **0.024%** | Multi-Match |
| **11 matches** | 37 | **0.002%** | Multi-Match |

- **Multi-Match Entities ($\ge 2$ matches):** **1,964,417 entities (89.02%)**
- **Singletons ($0$ matches):** **123,247 entities (5.58%)**
- **Single Match ($1$ match):** **119,157 entities (5.40%)**
- **Maximum Matches for a Single S1:** 11 matches
- **Mean Matches per S1:** 3.461 matches

### 4.2 Cross-Source Match Combinations
Examining the distribution of matches across Source 2 and Source 3:
- **Matches from both S2 and S3:** The dominant pattern (>80% of matched S1 entities), e.g.:
  - 1 from S2, 1 from S3: 269,681 entities (12.22%)
  - 1 from S2, 2 from S3: 251,224 entities (11.38%)
  - 2 from S2, 1 from S3: 223,054 entities (10.11%)
  - 2 from S2, 2 from S3: 208,159 entities (9.43%)
  - 1 from S2, 3 from S3: 140,393 entities (6.36%)
  - 2 from S2, 3 from S3: 116,076 entities (5.26%)
- **Matches only from S2 (0 from S3):** 117,240 entities (~5.3%)
- **Matches only from S3 (0 from S2):** 117,448 entities (~5.3%)
- **Zero matches from either (Singletons):** 123,247 entities (5.58%)

**Evaluation Metric Implication (Macro F0.5):**
- Singletons earn a score of **1.0** when correctly predicted as an empty string `""`, and **0.0** if even a single false-positive match is assigned.
- Because F0.5 rewards precision $2\times$ over recall ($\beta = 0.5$), false merges on singletons and false matches on multi-matches severely degrade the macro score.
- The model must have a calibrated threshold and explicit support for returning an empty match set.

---

## 5. Text Statistics & Noise Pattern Audit

### 5.1 Basic String Dimensions (Source 1)
- **Business Name Character Length:**
  - Mean: 24.0 characters | Median: 24 | 95th Percentile: 37 | Max: 71
- **Business Name Word Count:**
  - Mean: 3.5 words | Median: 4 | 95th Percentile: 5 | Max: 12
- **Business Address Character Length:**
  - Mean: 52.1 characters | Median: 41 | 95th Percentile: 103 | Max: 222
- **Business Address Word Count:**
  - Mean: 8.0 words | Median: 7 | 95th Percentile: 15 | Max: 38
- **Addresses with Numerical Identifiers:**
  - **95.16%** of all addresses contain at least one numerical identifier (house number, plot number, street number, or postal code).

### 5.2 Pairwise Similarity Distribution on Ground Truth Pairs
We evaluated ground-truth matching pairs to quantify name and address variations:

| Metric | Measured Value | Key Insight |
|---|---|---|
| **Exact Raw Name Match** | **3.03%** | Raw string comparison fails on 97% of true matches. |
| **Exact Lowercase Name Match** | **8.83%** | Simple lowercasing captures only 8.8%. |
| **Exact Suffix-Cleaned Name Match** | **44.64%** | Removing legal suffixes (Inc, LLC, Corp, Pvt Ltd) dramatically jumps exact matches by $5\times$! |
| **Mean Name Token Jaccard** | **0.6386** (Median: 0.6667) | Strong overall token overlap. |
| **Mean Norm-Name Token Jaccard** | **0.6814** (Median: 0.7500) | Normalization significantly concentrates similarity. |
| **Name Token Jaccard $\ge 0.5$** | **79.07%** | Nearly 80% of true matches share at least half their name tokens. |
| **Name Token Jaccard $\ge 0.2$** | **87.01%** | 87% share at least one significant name token. |
| **Exact Raw Address Match** | **1.64%** | Addresses are almost never formatted identically. |
| **Exact Lowercase Address Match** | **8.07%** | Casing accounts for a small portion of variation. |
| **Mean Address Token Jaccard** | **0.6051** (Median: 0.6429) | Address token overlap is robust across matches. |
| **Postal Code Agreement (when present)** | **88.68%** | Highly discriminative when available. |
| **Non-ASCII in Names** | **13.49%** | Significant proportion of Indian and French non-ASCII text. |

### 5.3 Deep Inspection: Why do ~13% of True Matches Have Low Name Jaccard?
Through targeted error audit of pairs with low name similarity (Jaccard $< 0.3$), we identified 4 distinct phenomena:

1. **Domain Names / URLs Used as Business Names:**
   - *Example:* S1: `Strategic Praetorian` vs. Target: `strategicpraetorian.com`
   - *Example:* S1: `Sakshi Sai Private Limited` vs. Target: `sakshisai.com`
   - *Example:* S1: `Gujarat Business Limited` vs. Target: `gujaratbusiness.com`
   - *Example:* S1: `Metropolitan Express Stores Inc` vs. Target: `metropolitanexpressstores.com`
   - *Solution:* Strip domain extensions (`.com`, `.org`, `.net`, `.in`, `.co.in`, etc.) and compare normalized alpha-strings.
2. **Social Media Handles / Truncated Nicknames:**
   - *Example:* S1: `Smart Raj Agro Pvt Ltd` vs. Target: `@smartraj`
   - *Solution:* Strip `@` prefix and perform substring / prefix token matching.
3. **Indic Regional Script Transliteration in India:**
   - *Example:* S1: `Universal Foods Pvt Ltd` vs. Target: `யுனிவர்சல் ஃபுட்ஸ் பிரைவேட் லிமிடெட்` (Tamil)
   - *Example:* S1: `Shakti Agro Limited` vs. Target: `ଶକ୍ତି ଆଗ୍ରୋ ଲିମିଟେଡ୍` (Odia)
   - *Example:* S1: `Prime Seven Impex Private Limited` vs. Target: `प्राइम सेवन इम्पेक्स प्राइवेट लिमिटेड` (Hindi / Devanagari)
   - *Example:* S1: `Dream Management Private Limited` vs. Target: `డ్రీమ్ మేనేజ్‌మెంట్ ప్రైవేట్ లిమిటెడ్` (Telugu)
   - *Crucial Finding:* In all these pairs, while the name is in regional script, **the address is in English with matching building/street numbers and street names** (e.g. `Plot No. 51, Aster Court, Vgp Golden Beach, Chennai`). **Address-based blocking directly recovers these pairs!**
4. **Acronyms and Significant Noise:**
   - *Example:* `Showalter-Templeton L.L.C.` vs. `Krystalle Showalter Templeton LLC`
   - Punctuation wrappers: `[[LLC]]`, `##2476`, `<< Team Ecole`.

---

## 6. Synthesis: Key Findings & Potential Risks

### 6.1 Key Findings
1. **100% Strict Country Invariant:** Zero cross-country matches exist. Exact country partitioning is completely safe, zero-loss, and drastically prunes the Cartesian space.
2. **89% Multi-Match Reality:** The vast majority of S1 entities correspond to 2 to 6 records across S2 and S3. Models must score candidate pairs independently without enforcing one-to-one constraints.
3. **5.58% Singletons:** Over 123,000 S1 records have zero matches. Any aggressive recall heuristic that forces matches will suffer major precision and macro F0.5 penalties.
4. **Legal Suffixes are the Primary Source of Name Mismatch:** Stripping legal suffixes raises exact name matches from 8.8% to 44.6%.
5. **Address is the Critical Safety Net:** For regional script transliterations and domain names, address numbers and street tokens provide the matching bridge.
6. **Country Shift in Test:** India is 46.75% of test (vs. 40.0% in train); France is 14.98% of test (vs. 0% in train). Hardcoded country dictionaries (e.g. US state lists or Indian PIN rules) would fail on France.

### 6.2 Potential Risks
- **Memory Exhaustion during Candidate Generation:** Comparing 1.73M test S1 records against 9.97M test S2/S3 records will run out of RAM if not chunked or implemented with sparse inverted indexes.
- **Transliteration Blind Spots:** Relying purely on business name blocking would miss the ~3-5% of Indian records transliterated into regional scripts.
- **Overfitting to US/India Specifics:** Rules tuned to US ZIP codes or Indian PIN codes will fail on French postal formats or French legal suffixes (SARL, SAS, SCI).
- **False-Positive Avalanche:** F0.5 heavily penalizes low precision ($2\times$ weight on precision). A low threshold will ruin the score.

---

## 7. Promising Blocking Strategies & Features

### 7.1 Multi-Pass Inverted-Index Blocking Strategy
To maximize candidate recall while minimizing candidate pair volume:
- **Hard Partition:** Block strictly within each `country`.
- **Pass 1 (Exact Normalized Name):** Lowercase, stripped punctuation, legal suffixes removed, concatenated domain equivalents.
- **Pass 2 (Rare Name Tokens):** Inverted index on informative name tokens (excluding stopwords and high-frequency noise words like "services", "center", "shop").
- **Pass 3 (Character 3-Grams / MinHash):** For names with typos or slight transliteration variations.
- **Pass 4 (Address Number + Street / City Token):** To catch transliterated names (where address is identical but name is in Tamil/Hindi/Telugu).

### 7.2 Promising Pairwise Features
- **Name Similarities:**
  - Token Jaccard, Token Overlap, Token Dice on raw and suffix-cleaned names.
  - Character 3-gram cosine similarity.
  - Normalized Levenshtein / Damerau-Levenshtein distance.
  - Exact match flags (raw, lowercase, suffix-cleaned, alphanumeric-only).
  - Prefix / Substring containment flag.
- **Address Similarities:**
  - Address token Jaccard and intersection size.
  - House / building number match flag.
  - Postal code match / mismatch / missing flag.
  - Exact address match flag.
- **Composite & Indicator Features:**
  - Source indicator (`is_s2` vs `is_s3`).
  - Missing address flag on candidate.
  - Length difference ratios for name and address.

---

## 8. Open Questions & Recommendations for Next Phases

### 8.1 Open Questions
1. *French Transliteration / Diacritics:* In France test records, accented characters (`é`, `è`, `ê`, `à`) appear frequently. Standardizing Unicode to NFKD and ASCII folding will ensure uniform matching.
2. *Candidate Count Ceiling:* The competition notes that "the approach that generates a smaller candidate set per Source 1 entity will be ranked higher". We should target an average of $\le 10$ to $20$ candidates per S1 while retaining $>95\%$ recall.

### 8.2 Recommended Next Experiments
1. **Phase 2 (Normalization Module):** Implement `src/normalize.py` with generic, multilingual text cleaning, legal suffix stripping (covering US, India, and France/international), and address component normalization.
2. **Phase 3 (Validation & Baseline Setup):** Construct a clean, stratified train/validation split (holding out 10-15% of S1 entities and their associated ground truth) and measure baseline exact-match F0.5.
3. **Phase 4 (Blocking Evaluation):** Benchmark the multi-pass blocking pipeline on the validation split, measuring candidate recall, candidate reduction ratio, and average candidates per S1.

---
*End of Phase 1 EDA Report.*
