# ds-projects
# Data Science Marketing Portfolio — Project Pack

**Projects:** Price Elasticity • Causal Inference • Predictive LTV/CLV • Marketing Strategy (MMM or Uplift)

---

## TL;DR
- Build **4 repos** (or 1 mono-repo with subprojects) that each tell a business story: price optimization, causal lift, LTV forecasting, and budget allocation/uplift targeting.  
- Use **config-driven pipelines**, **tests**, and **clear READMEs** (exec summary first).  
- Prefer **realistic decisions**: “what price now?”, “did the promo *cause* lift?”, “which customers to invest in?”, “where to spend next?”

---

## Project 1 — Price Elasticity of Demand (with promotion & seasonality)
**Business goal:** If I change price by ±5–15%, what happens to units, revenue, and margin? Should I discount this month?

**Data options**
- Public-ish: Kaggle “Retail Hero”/“Superstore” (needs cleaning), UCI Store Item Demand, M5 (Walmart) for structure, or **synthetic** panel: 100 SKUs × 52 weeks with price, promos, competitors, seasonality.

**Modeling paths**
- **Baseline:** log-log demand model with controls  
  $$\log(q_{i,t}) = \alpha_i + \beta \log(p_{i,t}) + \gamma^\top X_{i,t} + \delta_t + \epsilon$$
  where $X$ = promo flags, holidays, competitor price; $\delta_t$ = week fixed effects.
- **Hierarchical:** partial pooling by category/brand (e.g., Bayesian via PyMC) for stable elasticity.
- **Endogeneity guardrails:** instrument price with lagged competitor price or cost shocks (2SLS) if you can justify relevance/exclusion; else use control functions or proxy controls.
- **Simulation:** compute revenue/margin curves and price that maximizes $m \cdot q(p)$ under inventory/brand constraints.

**Outputs to show**
- SKU/category elasticity distribution, % with inelastic demand.
- Price–revenue–margin curves with “safe range” bands.
- “If we raise price 7% on inelastic SKUs A,B,C → +€X/month margin.”

**Repo extras**
```
src/elasticity/iv.py          # 2SLS helper
reporting/price_curves.py     # charts for curves & bands
tests/test_elasticity_spec.py # spec checks
notebooks/01_eda.ipynb        # short EDA
notebooks/02_spec_checks.ipynb# VIFs, partial residuals
```

---

## Project 2 — Causal Inference: Did the promo *cause* the lift?
**Business goal:** Estimate **incremental** sales from a promotion/campaign.

**Design choices**
- **Panel (regions/stores):**
  - **DiD:** treated vs control stores with staggered timing, store & time FE, pre-trend checks.
  - **Synthetic control:** one treated geo vs donor pool.
- **Observational customer-level:**
  - **Propensity Score Weighting/Matching** → ATE/ATT.
  - **Double ML / Causal Forest** for HTE (which segments benefited).
  - **Uplift modeling** (two-model or Class Transformation) for targeting.

**Key validity checks**
- Parallel trends (graphs + placebo tests).
- Covariate balance before/after weighting.
- Sensitivity analysis (Rosenbaum bounds or partial R² style).

**Outputs to show**
- “Promo increased conversions by **+3.2pp (CI: 1.1–5.3)** → **+€48k** net.”
- Uplift deciles chart → “Top 20% customers drive 65% of incremental profit.”
- A one-pager: **Decision** = “run promo only in segments {X,Y} next quarter.”

**Repo extras**
```
src/causal/design.py         # matching/weighting
src/causal/diagnostics.py    # pre-trend, balance plots
src/causal/uplift.py         # uplift learners
tests/test_balance.py        # SMD & KS checks
tests/test_placebo.py        # placebo effects
```

---

## Project 3 — Predictive LTV/CLV with Uncertainty & Actionability
**Business goal:** Forecast 12-month LTV for new/active customers to set **CAC caps**, **retention spend**, and **credit risk thresholds**.

**Data options**
- If no real data, simulate cohorts: signup date, orders, AOV, channel, country, tenure, churn hazard, promo sensitivity.

**Modeling options**
- **Buy-’Til-You-Die:** BG/NBD for transactions + Gamma-Gamma for spend; explainable baseline.
- **ML Survival / Hazard:** Cox or gradient boosting survival for churn; combine with spend model.
- **Direct regression:** XGBoost/LightGBM for 12-mo revenue with **calibration** (isotonic/Platt) and **prediction intervals** (Quantile loss).
- **Uncertainty:** show PI95 to inform risk-adjusted CAC (e.g., invest when P90 LTV > CAC).

**Outputs to show**
- LTV by channel/geo cohort curves; calibration plot; gain/lift from prioritizing top decile.
- “Cut CAC by €12 on channels where P50 LTV < CAC; reinvest €50k in paid search long-tail.”

**Repo extras**
```
src/clv/bgnbd.py
src/clv/gammagamma.py
src/clv/gbm_quantile.py
src/clv/calibration.py
reporting/ltv_cohorts.py     # cohort heatmaps, retention curves
tests/test_bgnbd_params.py
tests/test_calibration.py
```

---

## Project 4 — Marketing Strategy: MMM (budget allocation) **or** Uplift Targeting
Pick one (or do both small).

### Option A: Lightweight Marketing Mix Modeling (MMM)
**Goal:** Allocate monthly budget across channels to hit revenue with diminishing returns.

**Spec**
- Adstock + saturation (e.g., Hill/S-curve), control for price, seasonality, promotions.
- Regularized regression or Bayesian MMM (e.g., PyMC/Stan-like).
- Optimizer to reallocate budget under constraints (min/max per channel).

**Outputs**
- Response curves + ROAS by spend level, optimal budget plan, “what-if” sandbox.

### Option B: Uplift Targeting for Retention Campaign
**Goal:** Identify customers who change behavior *because* of the campaign (not always-buyers).

**Spec**
- Two-model uplift (treated vs control) or Meta-Learners (T-/S-/X-learner).
- Business guardrails: min audience size, contact cost, fairness/geos.
- KPI: **Qini** / **uplift@k** and incremental profit.

**Outputs**
- “Target deciles 8–10 → +€X incremental margin; avoid deciles 1–3 (harmers).”

---

## Shared Repo Pattern (use across all 4)
```
project/
  README.md
  pyproject.toml            # or requirements.txt
  src/project_name/         # package code
  configs/default.yaml      # data paths, model params
  data/                     # tiny synthetic or sampled data + schema.md
  notebooks/                # short, clean EDA/validation only
  scripts/                  # cli: train.py, evaluate.py, report.py
  tests/                    # pytest (data validation + core logic)
  reporting/                # chart scripts -> /reports/*.png
  reports/                  # auto-generated figures/tables
  Makefile                  # make setup|train|eval|report
  Dockerfile                # optional but nice
  LICENSE
```

**CLI example**
```bash
make setup
python scripts/train.py --config configs/default.yaml
python scripts/evaluate.py --config configs/default.yaml
python scripts/report.py --config configs/default.yaml
```

**README (exec-first)**
1. **Summary (3 bullets):** problem → action → impact  
2. **Data:** source, ethics, sampling, schema  
3. **Method:** model + assumptions; diagnostics links  
4. **Results:** key charts + business decision  
5. **Repro:** 3–5 commands; env/Docker  
6. **What’s next:** 3 scoped improvements

---

## Metrics & Diagnostics to Prioritize
- **Elasticity:** adjusted $R^2$, price coefficient stability across specs, IV relevance (first-stage F), counterfactual revenue gain.
- **Causal:** pre-trend plot, standardized mean differences (SMD), placebo tests, sensitivity bounds, uplift Qini.
- **CLV:** calibration (PIT / reliability), PI coverage, cohort MAPE, ranking lift vs random.
- **MMM/Uplift:** out-of-time fit, response curve plausibility, incremental profit not just ROAS.

---

## What to Show a CMO/CFO (one slide each project)
- **Elasticity:** “Price +7% on 18 inelastic SKUs → +€92k/mo margin, minimal volume loss.”
- **Causal:** “Geo promo caused +3.2pp conversion (CI), but only in young families—target those.”
- **CLV:** “Cut CAC by €12 on channels below P50 LTV; expand SEO +€40k where P90 > LTV cap.”
- **MMM/Uplift:** “Reallocate €120k from Display to Search/Email → +€180k/quarter incremental.”

---

## Stretch Ideas (to flex PyTorch / HF / Data Eng)
- **PyTorch:** implement elasticity via **hierarchical Bayesian** in Pyro or a small custom torch optimizer for price→revenue curves.
- **Hugging Face:** build a **data card** and publish synthetic dataset; use `datasets` for transforms/versioning.
- **Data Eng:** dbt models for marketing mart + Great Expectations checks; Prefect flow to run nightly; push metrics to a small dashboard.

---

## Quick Checklist
- [ ] 2–4 end-to-end, business-framed projects  
- [ ] Clean READMEs with run steps and results  
- [ ] Reproducible env + (optional) Docker  
- [ ] Tests + data validation  
- [ ] Screenshots/GIFs for stakeholders  
- [ ] No secrets/PII; synthetic samples provided  
- [ ] Pin repos + top-level portfolio README

---

## Next Step
I can scaffold all four projects (folders, `pyproject`, `Makefile`, `configs`, dummy data generators, and README templates) so you can push to GitHub and fill in the modeling code. Or we can start with **Elasticity + Causal** first (they pair beautifully: optimize price after you *prove* promo lift).

