SUMMARY:

The problem with existing models: most are trained on a single country or crop, ignore critical inputs like fertilizer usage, and offer zero explainability to decision-makers who actually need to act on the predictions.
Our approach: a fully reproducible, multi-country, multi-crop ML pipeline covering 1990–2023, fusing three open data sources 
- FAOSTAT (agricultural production),
- World Bank Climate Portal (temperature, precipitation, CO₂), 
-FAOSTAT Inputs (fertilizer data)

into a master dataset of 3,289 observations across 43 features.

The pipeline:
🔧 Full data cleaning: imputation, outlier detection (IQR + Z-score), standardization and temporal alignment

⚙️ Feature engineering: lagged yields (t-1, t-2, t-3), rolling statistics, climate anomalies, fertilizer intensity, temporal trend encoding

🤖 Three models benchmarked: Random Forest, HistGradientBoosting, and XGBoost

🏆 Final model: a stacked hybrid ensemble (HGB + XGBoost + Ridge meta-learner) achieving R² = 0.967, RMSE = 486 kg/ha, MAPE = 6.4%

🌐 Deployed as a Flask web app for real-time yield prediction by country, crop, and year

🔍 Interpretability via SHAP values, so the model doesn't just predict — it explains

The result: a decision-support tool that can help policymakers, agronomists, and researchers anticipate cereal production shifts before they become crises.

DEMO:

https://github.com/user-attachments/assets/3e979046-0f3b-4949-9da4-ed19d66460b3

BENCHMARKING: 
<img width="1639" height="503" alt="Capture d&#39;écran 2026-04-13 194313" src="https://github.com/user-attachments/assets/6b1fb49c-99f4-4a8f-8f8e-e88071ae840f" />
<img width="1620" height="507" alt="Capture d&#39;écran 2026-04-13 194256" src="https://github.com/user-attachments/assets/b1289958-84a9-4417-8932-fff84858053e" />
<img width="1011" height="894" alt="Capture d&#39;écran 2026-04-13 194225" src="https://github.com/user-attachments/assets/0550f378-b015-40e4-98a4-362c34bc9f43" />
<img width="1618" height="757" alt="Capture d&#39;écran 2026-04-13 194152" src="https://github.com/user-attachments/assets/fa853b45-e025-411b-9db9-5a8a3bb6d836" />
<img width="1510" height="756" alt="Capture d&#39;écran 2026-04-13 194130" src="https://github.com/user-attachments/assets/16f6b441-1481-4136-85b8-01d45bde3b42" />

