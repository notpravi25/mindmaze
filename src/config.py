"""
Configuration module for Amazon ML Challenge 2026: Business Entity Resolution.
All global paths, seeds, thresholds, and hyperparameters are centralized here.
"""

import os

# Base paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# Dataset File Paths
TRAIN_SOURCE1_PATH = os.path.join(TRAIN_DIR, "train_source1.tsv")
TRAIN_SOURCE2_PATH = os.path.join(TRAIN_DIR, "train_source2.tsv")
TRAIN_SOURCE3_PATH = os.path.join(TRAIN_DIR, "train_source3.tsv")
TRAIN_GROUND_TRUTH_PATH = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

TEST_SOURCE1_PATH = os.path.join(TEST_DIR, "test_source1.tsv")
TEST_SOURCE2_PATH = os.path.join(TEST_DIR, "test_source2.tsv")
TEST_SOURCE3_PATH = os.path.join(TEST_DIR, "test_source3.tsv")

# Split configuration
SPLITS_DIR = os.path.join(PROJECT_ROOT, "data", "splits")
VALIDATION_FRACTION = 0.20
RANDOM_SEED = 42

# Normalization settings
LEGAL_SUFFIXES = [
    # US / UK / International
    "inc", "incorporated", "llc", "l.l.c.", "corp", "corporation", 
    "co", "company", "ltd", "limited", "pvt", "private", "pvt ltd", 
    "private limited", "llp", "lp", "plc",
    # France / Europe
    "sarl", "s.a.r.l.", "sas", "s.a.s.", "sasu", "sci", "sa", "s.a.", 
    "eurl", "gie", "snc",
    # Common corporate entity descriptors
    "enterprises", "enterprise", "services", "solutions", 
    "technologies", "group", "holdings"
]

DOMAIN_EXTENSIONS = [
    ".com", ".org", ".net", ".in", ".co.in", ".co", ".io", 
    ".ai", ".fr", ".biz", ".info", ".edu", ".gov", ".us"
]
