Live Demo: https://team-nirikshan-ai.vercel.app/

## 📌 Executive Summary

The **MPLADS AI Monitoring, Anomaly Detection & Risk Intelligence Platform** (*MPLADS InsightGrid*) is an enterprise-grade analytical decision-support and early-warning intelligence system built for monitoring fund allocations, sanctions, expenditures, and physical completion under the **Member of Parliament Local Area Development Scheme (MPLADS)** governed by the **Ministry of Statistics and Programme Implementation (MoSPI)**.

Developed for the **Smart India Hackathon (SIH)**, the platform ingests and correlates **79,068+ sanctioned public works** across **6 official MoSPI datasets** (T1 to T6). It executes unsupervised machine learning (Isolation Forests), natural language processing (TF-IDF & Cosine Similarity), chronological rule engines, and spatial clustering to generate explainable, multi-dimensional risk scores for public auditors and district authorities.

### 🏛️ Core Governance Philosophy: Decision Support, Not Adjudication
The platform operates under strict **Responsible AI Governance Principles**:
- **Non-Accusatory Terminology**: Avoids non-adjudicated labels such as *"Fraud Detected"* or *"Corruption"*. Instead, it classifies observations as `Risk Indicator`, `Financial Outlier`, `Candidate Duplicate Pair`, `Compliance Evidence Gap`, or `Requires Field Audit`.
- **Explainable Evidence**: Every flagged work includes a detailed mathematical and contextual justification (*Risk Evidence Panel*), comparing project metrics against historical peer-group medians.
- **Evidence Gap Distinction**: A missing asset inspection photo or incomplete record is classified as an **evidence gap** requiring human verification rather than proof of illegality.

---

## 🏗️ System Architecture & End-to-End Pipeline

```
+---------------------------------------------------------------------------------------+
|                             RAW MoSPI DATASETS (T1 - T6)                              |
| T1: MP Allocations | T2: Calamity | T3: Recommended | T4: Sanctioned | T5: Completed | T6: Payments |
+---------------------------------------------------------------------------------------+
                                           │
                                           ▼
+---------------------------------------------------------------------------------------+
|                         PREPROCESSING & ETL INGESTION                                 |
|   • Regex Work ID Extractor: r"(WS/[A-Za-z0-9\-_]+/\d{4}-\d{4}/\d+)"                 |
|   • Fallback Deterministic Generator: WS/SANC/000001 (100% Non-Null ID Guarantee)     |
|   • Currency, DateTime Normalization & Multi-Table Relational Joining                 |
+---------------------------------------------------------------------------------------+
                                           │
                                           ▼
+---------------------------------------------------------------------------------------+
|                         MASTER ANALYTICAL DATASET (PARQUET)                           |
|           79,068 Sanctioned Base Records Joined across All Lifecycle Stages           |
+---------------------------------------------------------------------------------------+
        │                           │                           │                       │
        ▼                           ▼                           ▼                       ▼
+------------------+       +------------------+       +------------------+       +------------------+
|     MODULE 1     |       |     MODULE 2     |       |     MODULE 3     |       |     MODULE 4     |
| Financial        |       | Candidate        |       | Compliance       |       | Schedule &       |
| Anomaly Engine   |       | Duplicate Engine |       | Rule Matrix      |       | Progress Engine  |
| • Peer Medians   |       | • Spatial Scoping|       | • 45-Day Sanction|       | • Elapsed vs.    |
| • IQR Bounds     |       | • TF-IDF N-Grams |       | • Timeline Order |       |   Median Delay   |
| • Isolation Forest|      | • Cosine Sim     |       | • Photo Inspection|      | • Physical vs.   |
|   (Scikit-Learn) |       | • Split-Work Det.|       | • MoSPI 2023 Regs|       |   Financial Gap  |
+------------------+       +------------------+       +------------------+       +------------------+
        │                           │                           │                       │
        └───────────────────────────┴─────────────┬─────────────┴───────────────────────┘
                                                  │
                                                  ▼
                        +----------------------------------------------------+
                        |           MASTER COMPOSITE RISK ENGINE             |
                        | Composite Score = (0.28 * Fin) + (0.15 * Dup) +    |
                        |                   (0.42 * Comp) + (0.15 * Schedule)|
                        +----------------------------------------------------+
                                                  │
                                                  ▼
                        +----------------------------------------------------+
                        |            DATA SYNC & MLOps PIPELINE              |
                        | • 15-Day Automated MoSPI API Sync Runner           |
                        | • SHA-256 Snapshot Manager & Delta Diff Engine     |
                        | • MLflow Model Registry, Feature Drift & Versioning|
                        +----------------------------------------------------+
                                                  │
                                                  ▼
                        +----------------------------------------------------+
                        |            FASTAPI REST SERVER (Port 8000)         |
                        | /api/overview | /api/risk-monitor | /api/work-detail|
                        | /api/duplicate-candidates | /api/sync/* | /api/model|
                        +----------------------------------------------------+
                                                  │
                                                  ▼
                        +----------------------------------------------------+
                        |       REACT + TYPESCRIPT DASHBOARD (Port 3000)     |
                        | Executive View | Audit Queue | 360° Work Detail    |
                        | Duplicate Inspector | AI Copilot Chat Modal        |
                        +----------------------------------------------------+
```

---

## 💻 Complete Technology Stack

### 1. Backend & Machine Learning Core
| Technology | Version | Purpose |
| :--- | :--- | :--- |
| **Python** | `3.11+` | Primary programming language for backend and statistical pipelines |
| **FastAPI** | `^0.110.0` | High-performance asynchronous REST API framework |
| **Uvicorn** | `^0.28.0` | Production ASGI web server hosting the FastAPI service |
| **Mangum** | `^0.17.0` | Serverless ASGI adapter for AWS Lambda & Vercel deployment |
| **Pandas** | `^2.2.0` | High-throughput tabular data manipulation and joins |
| **NumPy** | `^1.26.0` | Optimized matrix and numerical operations |
| **PyArrow** | `^15.0.0` | Apache Parquet columnar read/write for millisecond feature access |
| **Scikit-Learn** | `^1.4.0` | Isolation Forest unsupervised anomaly detection & TF-IDF vectorization |
| **SciPy** | `^1.12.0` | Cosine similarity distance computation |
| **MLflow (Custom)** | `Built-in` | Model registry, experiment tracking, feature drift and run history |

### 2. Frontend & User Interface
| Technology | Version | Purpose |
| :--- | :--- | :--- |
| **React** | `^18.3.1` | Component-based single-page application framework |
| **TypeScript** | `^5.2.2` | Strict end-to-end type safety and schema validation |
| **Vite** | `^5.2.11` | Next-generation ultra-fast frontend build tooling and bundler |
| **Tailwind CSS** | `^3.4.0` | Utility-first CSS framework with custom dark/light theme tokens |
| **Recharts** | `^3.0.0` | Interactive SVG data visualization (log-scaled charts, state rankings) |
| **Lucide React** | `^0.378.0` | Clean, accessible iconography designed for public sector dashboards |

### 3. Cloud & Deployment
- **Vercel**: Edge-deployed React frontend and serverless API gateway via `api/index.py` and `vercel.json`.
- **Render**: Continuous background web service hosting FastAPI via `render.yaml`.

---

## 📊 Data Foundation & Ingestion (T1 – T6)

The platform standardizes and cross-references data across 6 primary MoSPI tables:

| Dataset ID | Dataset Name | Raw Record Count | Role in Analytical Pipeline |
| :--- | :--- | :--- | :--- |
| **T1** | `Allocated Limit for Honble MPs.csv` | 544 | MP fund ceiling baselines and constituency entitlement |
| **T2** | `Amount consented for Calamity.csv` | 13 | Emergency calamity fund approvals and diversion tracking |
| **T3** | `Works Recommended.csv` | 28,001 | Recommendation dates, proposed amounts, and initial scopes |
| **T4** | `Works Sanctioned.csv` | 79,068 | **Master Base**: Approved project budgets, sanction dates, and statuses |
| **T5** | `Works Completed.csv` | 12,001 | Completion timestamps, asset evidence photos, and final disbursals |
| **T6** | `Expenditure on Works.csv` | 45,001 | Itemized payment transactions, disbursement dates, and payment statuses |

### Relational Key Synthesis (`cleaner.py` & `master_builder.py`)
- Standardized regex pattern: `r"(WS/[A-Za-z0-9\-_]+/\d{4}-\d{4}/\d+)"`
- Auto-fallback generator: Assigns deterministic sequential IDs (`WS/SANC/000001`) to legacy records lacking explicit IDs.
- **Match Rates**: **100%** match rate between T6 expenditure records and T4 sanctioned base; **98.3%** match rate for T5 completed works.

---

## 🧠 Analytical Modules & Mathematical Models

### 1. Financial Anomaly Engine (`src/modules/financial_anomaly.py`)
Flags expenditure outliers by comparing individual project costs against peer-group baselines defined by `(State, Work Category)`:
1. **Peer Ratio Calculation**:
   $$\text{Peer Ratio} = \frac{\text{Sanction Budget}}{\text{Median Sanction}(\text{State, Category})}$$
2. **Interquartile Range (IQR) Outlier Fences**:
   $$\text{IQR} = Q_3 - Q_1$$
   $$\text{Upper Fence} = Q_3 + (1.5 \times \text{IQR})$$
3. **Isolation Forest Machine Learning**:
   An unsupervised ensemble of isolation trees trained on:
   $$\vec{X} = \left[ \text{sanction\_amount}, \, \text{effective\_expenditure}, \, \text{peer\_ratio}, \, \text{max\_payment}, \, \text{single\_payment\_share} \right]$$
4. **Unit-Price & Material Context**: If descriptions specify physical dimensions (e.g., *2.5 km road*, *500 bags cement*), unit rates are computed and compared against state PWD benchmark rates (`nirikshan_material_price_benchmarks.csv`).

### 2. Candidate Duplicate & Split-Work NLP Engine (`src/modules/duplicate_detection.py`)
Detects duplicate recommendations or intentionally split tenders designed to bypass financial approval thresholds:
1. **Spatial & Categorical Partitioning**: Pairwise comparisons are restricted within `(State, Constituency)` blocks to avoid $O(N^2)$ cross-national combinatorial explosion.
2. **Text Normalization**: Strips administrative prefixes, normalizes Hindi/English transliterations, and removes punctuation.
3. **TF-IDF N-Gram Vectorization**: Extracts unigram and bigram tokens $(1, 2)$ over combined recommended + sanctioned descriptions.
4. **Cosine Similarity Formulation**:
   $$\text{Similarity}(\mathbf{A}, \mathbf{B}) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\|_2 \|\mathbf{B}\|_2} \times 100$$
5. **Split-Work Heuristics**: Flags pairs with $\ge 70\%$ text similarity sanctioned within **180 days** of each other with sequential or proximate Work IDs.

### 3. Compliance Matrix Engine (`src/modules/compliance_engine.py`)
Enforces deterministic validation rules anchored directly to the **2023 MoSPI MPLADS Guidelines**:
- **`RULE_COMP_01` (Missing Site Evidence Photo)**: Completed work without photographic verification (+40 points).
- **`RULE_COMP_02` (Chronological Inversion)**: Sanction date earlier than recommendation date, or completion date before sanction date (+50 points).
- **`RULE_COMP_03` (45-Day Guideline Violation)**: Implementing District Authority (IDA) failed to sanction/reject work within 45 days of MP recommendation (+30 points, MoSPI Guideline 3.2.4).
- **`RULE_COMP_04` (Zero Disbursal on Completed Work)**: Work marked as completed but total recorded expenditure $\le 0$ (+25 points).
- **`RULE_COMP_05` (Prohibited Category Scan)**: Scans description for unauthorized categories (commercial assets, religious structures, private land improvements, swagat dwar/welcome gates).

### 4. Schedule & Completion Delay Engine (`src/modules/delay_risk.py`)
- Calculates historical median completion periods for each work category.
- Measures the **Progress-to-Expenditure Gap**: Identifies projects where financial disbursal exceeds 80% while elapsed time exceeds 150% of category benchmarks.

### 5. Master Composite Risk Score (`src/risk/composite_risk_engine.py`)
Combines all risk dimensions using a **compliance-first priority model**:
$$\text{Composite Risk Score} = (0.28 \times S_{\text{financial}}) + (0.15 \times S_{\text{duplicate}}) + (0.42 \times S_{\text{compliance}}) + (0.15 \times S_{\text{schedule}})$$

#### Risk Severity Classification
- **LOW** (`0.0 – 34.9`): Normal baseline; standard operational variance.
- **MEDIUM** (`35.0 – 64.9`): Moderate risk; scheduled for periodic routine review.
- **HIGH** (`65.0 – 84.9`): Priority audit queue; statistical or procedural anomalies detected.
- **CRITICAL** (`85.0 – 100.0`): Multi-signal anomaly; immediate supervisory field inspection required.

---

## 🤖 AI Copilot (InsightGrid Assistant Modal)

The web dashboard features an interactive **AI Copilot** (`AiAssistantModal.tsx`):
- **Live Context Injection**: Automatically ingests portfolio health statistics (total works, high-risk works, financial outliers, candidate duplicate pairs).
- **Natural Language Query Routing**:
  - *"Show high risk works"* $\rightarrow$ Provides deep links directly to the filterable Risk Monitor queue.
  - *"Find financial disbursal mismatches"* $\rightarrow$ Filters works with high peer deviation ratios.
  - *"Check duplicate works in Uttar Pradesh"* $\rightarrow$ Opens the Duplicate Inspector pre-scoped to target constituencies.
- **Explainable Work Summaries**: Summarizes why any specific `work_id` was flagged without technical jargon.

---

## 🔄 Automated Data Synchronization & MLOps Pipeline

Located in `src/data/sync/`, the platform includes an enterprise data synchronization and model retraining suite:
1. **15-Day Automated Sync Runner (`sync_runner.py`)**: Asynchronously queries official MoSPI REST endpoints.
2. **Delta Engine (`delta_engine.py`)**: Compares incoming records against active Parquet stores, computing exact `NEW`, `MODIFIED`, and `REMOVED` record diffs.
3. **SHA-256 Snapshot Manager (`snapshot_manager.py`)**: Takes cryptographically verified snapshots before committing any dataset update, enabling zero-downtime rollbacks.
4. **Automated Retraining Manager (`training_manager.py`)**: Retrains Isolation Forest models and re-computes TF-IDF matrices upon confirmed sync commits.
5. **Model Registry (`mlflow_tracker.py`)**: Local MLflow-compatible JSON registry tracking model versions, anomaly rates, and parameter drift.

---

## 📁 Repository Directory Structure

```text
sih_new/testai-chatBot (1)/testai-chatBot/
├── api/
│   └── index.py                     # Vercel Serverless Mangum entry point
├── data/
│   ├── raw/                         # Raw MoSPI CSV files (T1 - T6)
│   ├── processed/                   # Cleaned, type-validated Parquet datasets
│   ├── features/                    # Analytical feature matrices & risk score Parquet files
│   ├── reference/                   # Sector cost reference & material benchmark rates
│   ├── sync_history.json            # Historical record of all sync jobs
│   └── mlflow_model_registry.json   # Model versions, metrics, and training timestamps
├── docs/
│   ├── MPLADS_PLATFORM_GUIDE.md     # Detailed user and operator manual
│   ├── MPLADS_CODEBASE_AUDIT.md     # In-depth architectural audit
│   ├── data_dictionary.md           # Schema, columns, and field definitions for T1-T6
│   └── MPLADS_Platform_User_and_Model_Guide.pdf
├── frontend/                        # React + TypeScript Visual Intelligence Dashboard
│   ├── src/
│   │   ├── components/              # Reusable UI cards, tables, GIS maps, layout shells
│   │   │   ├── AiAssistantModal.tsx # Interactive AI Copilot assistant modal
│   │   │   ├── layout/              # Sidebar, Topbar, Header navigation
│   │   │   └── risk/                # RiskBadge, RiskGauge, RiskEvidencePanel
│   │   ├── pages/                   # 12 Dedicated Dashboard Views
│   │   │   ├── OverviewPage.tsx     # National executive summary & KPIs
│   │   │   ├── RiskMonitorPage.tsx  # Filterable priority audit queue
│   │   │   ├── ProjectDetailPage.tsx# 360° deep-dive audit view for any work
│   │   │   ├── DuplicateInspectorPage.tsx # Side-by-side duplicate comparison
│   │   │   ├── FinancialAnalyticsPage.tsx # Outlier distributions & peer medians
│   │   │   ├── FinancialBenchmarkPage.tsx # Sector-level unit price benchmarks
│   │   │   ├── ComplianceMonitorPage.tsx  # MoSPI guideline violation matrix
│   │   │   ├── ScheduleProgressPage.tsx   # Delay tracking & progress mismatches
│   │   │   ├── MpIntelligencePage.tsx     # MP-level allocation & utilization
│   │   │   ├── StateRiskAnalyticsPage.tsx # State-wise risk concentration
│   │   │   ├── DataSyncPage.tsx           # MoSPI API live sync & delta diff inspector
│   │   │   └── ModelMonitoringPage.tsx    # MLflow model metrics & drift monitor
│   │   ├── services/api.ts          # Strongly typed client API service
│   │   └── types/index.ts           # Comprehensive TypeScript schema definitions
│   ├── package.json
│   └── vite.config.ts
├── render.yaml                      # Render cloud deployment specification
├── requirements.txt                 # Python backend dependencies
├── vercel.json                      # Vercel deployment routing configuration
└── src/
    ├── backend/
    │   └── app.py                   # FastAPI REST server with Lazy Parquet Indexing
    ├── data/
    │   ├── master_builder.py        # Joins T1-T6 into master_analytical.parquet
    │   ├── build_financial_benchmarks.py # Generates peer-group median tables
    │   └── sync/                    # Sync runner, delta engine, snapshots, training
    ├── modules/                     # Core Analytical Engines:
    │   ├── financial_anomaly.py     # Isolation Forest & peer median ratios
    │   ├── duplicate_detection.py   # TF-IDF & Cosine text similarity
    │   ├── compliance_engine.py     # 2023 MoSPI guideline compliance rules
    │   ├── delay_risk.py            # Completion timelines & progress gap analysis
    │   ├── work_classifier.py       # Rule-based NLP work categorization
    │   ├── sector_classifier.py     # Sector mapping & regional cost multipliers
    │   └── material_context.py      # Cement/steel/brick quantity extraction
    ├── preprocessing/
    │   ├── cleaner.py               # Regex Work ID extraction & type cleaning
    │   └── data_profiler.py         # Summary statistics and missing value audit
    ├── risk/
    │   └── composite_risk_engine.py # Master multi-signal risk synthesizer
    ├── utils/
    │   └── mlflow_tracker.py        # Custom MLflow run tracker & metadata logger
    └── run_pipeline.py              # Master pipeline orchestrator script
```

---

## 🌐 API Reference (FastAPI REST Server)

The FastAPI server runs by default on `http://127.0.0.1:8000` and provides interactive Swagger documentation at `http://127.0.0.1:8000/docs`.

| Endpoint | Method | Key Parameters | Description |
| :--- | :---: | :--- | :--- |
| `/api/health` | `GET` | — | Health check, active record count (79,068), and memory status. |
| `/api/overview` | `GET` | — | Portfolio KPIs: total allocated, sanctioned, disbursed, and risk tier counts. |
| `/api/risk-monitor` | `GET` | `state`, `constituency`, `category`, `severity`, `search`, `page`, `limit` | Paginated priority audit queue with multi-column filtering. |
| `/api/work-detail` | `GET` | `work_id` (Query or Path `{work_id:path}`) | Complete 360° work audit profile, evidence panel, and duplicate links. |
| `/api/duplicate-candidates` | `GET` | `state`, `constituency`, `min_similarity`, `page`, `limit` | Pairwise candidate duplicate records with match scores. |
| `/api/financial-benchmarks` | `GET` | `state`, `category` | Historical peer median costs and unit price reference tables. |
| `/api/state-risk-summary` | `GET` | `state` | State-level risk breakdown, dominant signals, and expenditure totals. |
| `/api/mp-intelligence` | `GET` | `state`, `constituency`, `mp_name` | MP-level work counts, fund utilization rate, and flagged works. |
| `/api/schedule-risk` | `GET` | `state`, `category`, `page`, `limit` | Delayed projects, overdue durations, and progress gap metrics. |
| `/api/filters` | `GET` | — | Dropdown filter options (states, constituencies, categories, MPs). |
| `/api/sync/status` | `GET` | — | Live MoSPI sync status, active snapshot ID, and next scheduled sync. |
| `/api/sync/preview-diff` | `GET` | — | Stages incoming REST data and returns record-level NEW/MODIFIED/REMOVED deltas. |
| `/api/sync/commit-diff` | `POST` | — | Promotes reviewed staged data, generates a snapshot, and queues retraining. |
| `/api/sync/training-status`| `GET` | — | Background model retraining progress, feature drift, and validation results. |
| `/api/model/status` | `GET` | — | Active MLflow model versions, training timestamps, and anomaly rates. |

---

## 🚀 Installation & Local Setup

### Prerequisites
- **Python**: Version `3.11` or higher
- **Node.js**: Version `18.0.0` or higher
- **Git**

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd sih_new/testai-chatBot\ \(1\)/testai-chatBot
```

### Step 2: Backend Setup & Environment
```bash
# 1. Create a Python virtual environment
python -m venv .venv

# 2. Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (Command Prompt):
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# 3. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Run the Analytical Data Pipeline (Initial Run)
```bash
# Executes data cleaning, master building, ML anomaly detection,
# TF-IDF duplicate matching, compliance scoring, and composite synthesis:
python src/run_pipeline.py
```
*(Execution typically completes in ~15–25 seconds and generates all optimized Parquet files in `data/features/`)*.

### Step 4: Start the FastAPI Backend
```bash
python -m uvicorn src.backend.app:app --host 127.0.0.1 --port 8000 --reload
```
- API Base: `http://127.0.0.1:8000`
- Interactive Swagger Docs: `http://127.0.0.1:8000/docs`

### Step 5: Start the React Frontend Dashboard
Open a new terminal window:
```bash
cd frontend
npm install
npm run dev
```
- Frontend Application: `http://localhost:3000`

---

## ☁️ Deployment Guide

### Vercel (Frontend & Serverless API)
The project includes pre-configured `vercel.json` and `api/index.py` using Mangum:
```bash
# From repository root:
vercel --prod
```
- Frontend build command: `cd frontend && npm ci --legacy-peer-deps && npm run build`
- Output directory: `frontend/dist`
- API routing: Rewrites `/api/*` requests to `api/index.py`.

### Render (FastAPI Web Service)
The project includes `render.yaml`:
```yaml
services:
  - type: web
    name: mplads-backend
    env: python
    plan: free
    buildCommand: "pip install -r requirements.txt"
    startCommand: "uvicorn src.backend.app:app --host 0.0.0.0 --port 8000"
    envVars:
      - key: PYTHON_VERSION
        value: "3.11"
```

---

## 🎯 Smart India Hackathon (SIH) Jury Presentation Flow

Follow this structured **3-minute live presentation sequence**:

1. **National Executive Overview (`OverviewPage.tsx`)**:
   - Highlight the total portfolio: **79,068 works**, **₹4,168 Cr sanctioned budget**, and the **4,951 cases flagged for review**.
   - Show the logarithmic risk distribution bar chart highlighting the separation between low baseline and priority cases.
2. **Operational Review Queue (`RiskMonitorPage.tsx`)**:
   - Filter severity to `HIGH (65-84 Score)`.
   - Demonstrate the multi-column sort (Financial, Duplicate, Compliance, Schedule).
3. **Deep-Dive Audit View (`ProjectDetailPage.tsx`)**:
   - Click **Inspect** on a high-risk work record (e.g., `WS/MP792/2024-2025/176431`).
   - Showcase the **Risk Evidence Panel**: Show how the budget (₹85.94 Lakh) is **$37.1\times$ higher than the peer group median** (₹2.32 Lakh) and in the 99.9th percentile.
4. **Duplicate & Split-Work Inspector (`DuplicateInspectorPage.tsx`)**:
   - Present candidate duplicate pairs with $>80\%$ cosine text similarity within the same constituency.
5. **AI Copilot & Data Sync Engine**:
   - Open the **AI Assistant Copilot modal** and ask: *"Show works with critical composite risk scores"*.
   - Navigate to **Data Sync & System Status** to show automated 15-day MoSPI REST synchronization, delta diffing, and SHA-256 snapshotting.

---

## 🔒 Limitations & Ethical Guardrails

1. **Text Normalization Limitations**: Free-text work descriptions in regional government databases may contain spelling variations or incomplete specifications.
2. **Reference Costs**: Sector cost ranges serve as comparative benchmarks, not statutory spending caps.
3. **Field Audit Requirement**: All platform outputs represent mathematical signals to guide inspection priorities. Final administrative or legal decisions must be confirmed through field audits, physical site inspections, and verification of voucher documentation.

---
