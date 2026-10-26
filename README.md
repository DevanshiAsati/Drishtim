# Intelligent Mule Account Detection and Financial Fraud Investigation Platform

An MVP that detects suspected mule bank accounts from transaction behaviour, traces
fund flows through a graph, correlates cybercrime complaints with payments, and
presents everything in an investigation dashboard with explainable risk scores.

## Quick start

```bash
pip install -r requirements.txt

# 1. Generate synthetic data (writes data/*.csv)
python src/data_generator.py

# 2. Run the analysis from the command line
python src/pipeline.py

# 3. Launch the investigation dashboard
streamlit run app/dashboard.py

# 4. Run the tests
PYTHONPATH=src pytest -q tests
```

## Project structure

```
mule_detection/
├── app/
│   └── dashboard.py          Streamlit investigation dashboard
├── src/
│   ├── data_generator.py     Synthetic accounts, transactions, complaints with injected mule rings
│   ├── scoring.py            Behavioural features and explainable weighted risk score
│   ├── graph_analysis.py     Directed graph, money-trail tracing, clusters, intermediaries
│   ├── complaint_correlation.py  Complaint-to-transaction matching and incident grouping
│   └── pipeline.py           Orchestrates all modules into one AnalysisResult
├── tests/
│   └── test_pipeline.py      Tests for generator, scoring, graph and correlation
├── data/                     Generated CSVs (created by data_generator.py)
├── reports/                  Place for exported case reports
└── requirements.txt
```

## How it maps to the problem statement

| Requirement | Implementation |
|---|---|
| Transaction analysis | `scoring.compute_features`: fan-in, pass-through ratio, rapid forwarding, counterparty diversity, dormant activation |
| Mule account detection | `scoring.score_accounts`: 0–100 risk score, risk bands, top-3 indicators per account |
| Money trail analysis | `graph_analysis.trace_money_trail`: ranked fund-flow paths from a victim account |
| Complaint correlation | `complaint_correlation.correlate_complaints` and `complaint_incident_groups` |
| Graph-based investigation | `graph_analysis.detect_clusters`, `find_intermediaries`, and the network view in the dashboard |
| Investigation dashboard | Ranked alerts, network view, money trail with timeline, complaint groups, per-account explanations, CSV case export |

## Scoring model

Each account receives six features, each normalised to [0, 1]:

| Feature | Weight | Meaning |
|---|---|---|
| `pass_through_ratio` | 0.25 | Share of inflow forwarded out (only counts at 60%+) |
| `fan_in_ratio` | 0.20 | Distinct senders, ramped from 6 to 20 |
| `rapid_forward_score` | 0.20 | Median hours between receiving and forwarding |
| `complaint_exposure` | 0.15 | Distinct victims who reported the account |
| `counterparty_diversity` | 0.10 | Distinct counterparties, ramped from 15 to 50 |
| `dormant_activation` | 0.10 | Recently opened or sparse-history account |

The final score is the weighted sum scaled to 0–100. Accounts inside a suspicious
graph cluster with at least two other high-risk accounts receive a +10 boost.
Bands: Low below 35, Medium 35–55, High 55 and above.

Every score comes with the top three contributing indicators and their point values,
shown in the Explanations tab.

## Limitations and next steps

- **Synthetic data only.** Thresholds and ramps were tuned on the generated dataset
  to separate injected mule rings from normal activity. Real deployments need
  calibration on institutional data and a labelled validation set.
- **Rule-based scoring.** The design allows adding an Isolation Forest or
  gradient-boosted model as an additional feature. This is not yet included.
- **Rapid-forward computation** is a per-account loop and will be slow on
  very large transaction tables. Vectorising it or using a groupby-merge-asof
  approach is the first optimisation to make.
- **Money-trail search** enumerates simple paths up to depth 4. This is fine for
  thousands of edges but needs bounding or a best-first search at larger scale.
- **Complaint matching** uses exact counterparty pairs, amount tolerance and a
  7-day window. Fuzzy matching on account aliases is not implemented.
- **Privacy and compliance.** Any production use must address data protection law,
  access control and audit logging for investigator actions.
