# Dynamic Pricing for Perishable Inventory — Pipeline + Dashboard

FSE 570 Capstone, Team Missouri.

## Folder layout

```
Capstone_Missouri/
├── .gitignore
├── Datasets/                                    ← raw data (not in git)
│   ├── perishable_goods_management.csv
│   └── dunnhumby_The-Complete-Journey/...
├── code/                                        ← THIS folder
│   ├── config.py                 paths + hyperparameters
│   ├── data_loading.py           load + 3-way split
│   ├── features.py               lag/rolling/calendar features
│   ├── demand_forecast.py        Stage 1 — LGB / Prophet / HW
│   ├── elasticity.py             Stage 2 — Dunnhumby OOS
│   ├── spoilage.py               Stage 3 — 4-classifier comparison
│   ├── optimizer.py              Stage 4 — batch_simulate
│   ├── simulation.py             Stage 4 — KPIs / breakdowns
│   ├── sensitivity.py            Stage 5 — high-spoil / elas / stability
│   ├── run_pipeline.py           orchestrator (CLI)
│   ├── pipeline_notebook.ipynb   orchestrator (notebook)
│   ├── save_whatif_files.py      ← run after pipeline; produces dashboard inputs
│   ├── dashboard.py              ← Streamlit data-storytelling app
│   ├── requirements.txt
│   ├── environment.yml
│   └── README.md
├── outputs/                                     ← created by run_pipeline.py
└── models/                                      ← created by run_pipeline.py
```

## 1. Set up the env (one time)

```powershell
cd C:\Users\Bhavana\Downloads\Capstone_Missouri\Capstone_Missouri\code
conda env create -f environment.yml
conda activate dynamic_pricing
```

If conda hangs on Prophet for >10 min:

```powershell
conda create -n dynamic_pricing python=3.11 -y
conda activate dynamic_pricing
conda install -c conda-forge prophet -y
pip install -r requirements.txt
```

Register the env's kernel for Jupyter (only needed once):

```powershell
conda install -n dynamic_pricing ipykernel -y
python -m ipykernel install --user --name dynamic_pricing --display-name "Python 3 (dynamic_pricing)"
```

## 2. Run the pipeline

**Option A — script:**

```powershell
conda activate dynamic_pricing
cd C:\Users\Bhavana\Downloads\Capstone_Missouri\Capstone_Missouri\code
python run_pipeline.py --skip-prophet
```

(Drop `--skip-prophet` for a final run that includes Prophet in the comparison table; adds ~5 minutes.)

**Option B — Jupyter:**

```powershell
conda activate dynamic_pricing
cd C:\Users\Bhavana\Downloads\Capstone_Missouri\Capstone_Missouri\code
jupyter notebook
```

Open `pipeline_notebook.ipynb`, switch kernel to **Python 3 (dynamic_pricing)** in the top-right corner, then **Kernel → Restart & Run All**.

## 3. Generate the dashboard inputs

The dashboard's What-If section needs two extra files. After the pipeline finishes:

```powershell
python save_whatif_files.py
```

Takes ~30 seconds (re-runs the LightGBM holdout forecast).

## 4. Run the dashboard locally

```powershell
streamlit run dashboard.py
```

Opens automatically at http://localhost:8501. 13 sections in the sidebar; the **What-if explorer** lets you adjust elasticity, near-expiry threshold, near-expiry elasticity, margin floor, and discount grid — and re-runs the optimizer on a 500-row holdout sample side-by-side with the project baseline.

## 5. Deploy to Streamlit Community Cloud

Streamlit Cloud is free and the natural fit (Vercel doesn't run Python).

### a. Set up GitHub

```powershell
cd C:\Users\Bhavana\Downloads\Capstone_Missouri\Capstone_Missouri
git init
git add .
git commit -m "Capstone — dynamic pricing pipeline + dashboard"
```

Push to a repo on github.com (private is fine if Streamlit Cloud has access). Note: `.gitignore` excludes `Datasets/`, the legacy `Code Files/` folder, and the .docx/.pdf/.mp4/.pptx deliverables. **`outputs/` and `models/` are intentionally committed** so the deployed dashboard has data to read.

```powershell
git remote add origin https://github.com/YOUR_USERNAME/capstone-dynamic-pricing.git
git branch -M main
git push -u origin main
```

### b. Create the Streamlit Cloud app

1. Go to https://share.streamlit.io
2. Click **New app**
3. Connect your GitHub account
4. Select the repo
5. **Branch**: `main`
6. **Main file path**: `code/dashboard.py`
7. **App URL**: pick a slug, e.g. `team-missouri-dynamic-pricing`
8. Under **Advanced settings**:
   - **Python version**: 3.11
9. Click **Deploy**

First deploy takes 3–5 min. Future pushes to `main` auto-redeploy.

### c. If the build fails

Common causes:

- **Prophet won't install on Cloud**. Streamlit Cloud uses pip (not conda), and Prophet is fragile. Easiest fix: pin `prophet>=1.1.5` in `requirements.txt` (already done) and let it try. If it still fails, remove Prophet from `requirements.txt` — the dashboard doesn't need it (only `run_pipeline.py` does, and that runs on your laptop, not on Cloud).
- **Memory limits** (1 GB free tier). The dashboard loads `sim_results.csv` (~2 MB) and `spoilage_model.pkl` (~1 MB) — well under the limit. If Cloud OOMs anyway, drop the `cat` parameter from the LightGBM model or sample down `sim_results.csv` to 5,000 rows.

## What the pipeline produces

In `../outputs/`:

| File | Used by | Description |
|---|---|---|
| `manifest.json` | dashboard | Run metadata (winners, elasticities, waste fraction) |
| `strategy_kpi.csv` | dashboard | Revenue / profit / waste / sell-through per strategy |
| `strategy_delta_table.csv` | dashboard | Side-by-side delta table |
| `category_profit.csv` | dashboard | Profit by category × strategy |
| `near_expiry_buckets.csv` | dashboard | Discount + profit/unit by days-to-expiry |
| `daily_profit_trend.csv` | dashboard | Daily profit per strategy |
| `high_spoilage_kpi.csv` | dashboard | KPIs on `days_until_expiry ≤ 3` subset |
| `elasticity_sensitivity.csv` | dashboard | ±50% elasticity test, full + non-near-expiry |
| `discount_distribution.csv` | dashboard | Aggregate discount distribution |
| `category_mean_discount.csv` | dashboard | Mean recommended discount by category |
| `optimizer_stability_sample.csv` | dashboard | 1k-row optimizer recommendations |
| `elasticity_estimates.csv` | dashboard | Elasticity + OOS R² per department |
| `demand_model_comparison.csv` | dashboard | LGB / Prophet / HW val metrics |
| `spoilage_model_comparison.csv` | dashboard | LR / RF / XGB / LGB val metrics |
| `sim_results.csv` | dashboard | Per-row simulation results (~6.7k × 3 strategies) |
| `whatif_sample.csv` | dashboard what-if | 500-row holdout sample for live re-optimization |
| `sku_demand_forecasts.csv` | dashboard what-if | Per-SKU per-day demand forecasts |

In `../models/`:

| File | Description |
|---|---|
| `spoilage_model.pkl` | Pickled spoilage model + train_cols + waste_frac. Loaded by the what-if to score what-if scenarios. |

## Known caveats (deliberately not "fixed" — flagged in the report)

1. **Elasticity sensitivity ±50% on the full sample produces near-identical profits** because 42% of rows hit the −2.0 near-expiry override. The pipeline now also produces a `NON_NEAR_EXPIRY_ONLY` variant — that's the test that actually evaluates whether the regular elasticity coefficient is doing work in the optimizer.
2. **Ready_to_Eat stays negative under Dynamic** (–$158K). Likely structural (thin margin × high spoilage). The category-level breakdown shows it explicitly.
3. **Spoilage target is binary** (`units_wasted > 0`). The empirical waste fraction calibrates this into an expected quantity, but the model itself does not predict the magnitude of waste.
4. **OOS R² for elasticity is negative on PRODUCE/MEAT** because the OOS test deliberately strips store + week fixed effects. The elasticity *coefficient* is stable across windows; that's what the validation actually demonstrates. Frame it carefully in the report.
