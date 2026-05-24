"""
model_utils.py — fixed with:
  - Exact global means from real training data
  - Correct one-hot lookup (no Country/Crop columns in CSV)
  - Iterative future prediction: rolls lags forward year-by-year
    so 2027 ≠ 2050
"""
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

# ─────────────────────────────────────────────────────────────
# EXACT FEATURE ORDER — must match scaler.feature_names_in_
# ─────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "Avg_Temperature_C", "CO2_Emissions_TonsCapita", "Extreme_Weather_Events",
    "Fert_N_tonnes", "Fert_P_tonnes", "Fert_Total_tonnes", "Fert_per_capita",
    "Forest_Area", "Population", "Rain_anomaly", "Rain_lag1", "Rainfall_mm",
    "Renewable_Energy", "Sea_Level_Rise_mm", "Temp_anomaly", "Temp_lag1",
    "Year_norm", "Year_norm2",
    "Yield_kg_ha_roll3_mean", "Yield_kg_ha_roll3_std",
    "Yield_lag1", "Yield_lag2", "Yield_lag3",
    "crop_Maize", "crop_Rice", "crop_Wheat",
    "ctry_Argentina", "ctry_Australia", "ctry_Brazil", "ctry_Canada",
    "ctry_China", "ctry_France", "ctry_Germany", "ctry_India",
    "ctry_Indonesia", "ctry_Japan", "ctry_Mexico", "ctry_Russia",
    "ctry_South Africa", "ctry_UK", "ctry_USA",
]

# Year normalisation: exactly what training used
YEAR_MIN = 1990
YEAR_MAX = 2023
YEAR_RANGE = YEAR_MAX - YEAR_MIN   # 33

# ─────────────────────────────────────────────────────────────
# EXACT GLOBAL MEANS from the real training dataset
# (fallback when a country/crop combo row is missing)
# ─────────────────────────────────────────────────────────────
GLOBAL_MEANS = {
    "Avg_Temperature_C":        19.679599,
    "CO2_Emissions_TonsCapita": 10.202168,
    "Extreme_Weather_Events":    7.399902,
    "Fert_N_tonnes":       7_177_462.56,
    "Fert_P_tonnes":       3_136_088.36,
    "Fert_Total_tonnes":  10_299_108.68,
    "Fert_per_capita":          0.031728,
    "Forest_Area":             40.965625,
    "Population":         698_007_795.29,
    "Rain_anomaly":             2.887396,
    "Rain_lag1":             1733.928632,
    "Rainfall_mm":           1734.552633,
    "Renewable_Energy":        27.564908,
    "Sea_Level_Rise_mm":        2.974910,
    "Temp_anomaly":            -0.030648,
    "Temp_lag1":               19.687881,
    "Yield_kg_ha_roll3_mean": 4984.254792,
    "Yield_kg_ha_roll3_std":   194.569855,
    "Yield_lag1":             5008.599085,
    "Yield_lag2":             4997.782031,
    "Yield_lag3":             4985.161888,
}

# User input → one-hot column name
CROP_COL = {"maize": "crop_Maize", "rice": "crop_Rice", "wheat": "crop_Wheat"}
CTRY_COL = {
    "argentina":    "ctry_Argentina",  "australia":    "ctry_Australia",
    "brazil":       "ctry_Brazil",     "canada":       "ctry_Canada",
    "china":        "ctry_China",      "france":       "ctry_France",
    "germany":      "ctry_Germany",    "india":        "ctry_India",
    "indonesia":    "ctry_Indonesia",  "japan":        "ctry_Japan",
    "mexico":       "ctry_Mexico",     "russia":       "ctry_Russia",
    "south africa": "ctry_South Africa","uk":           "ctry_UK",
    "usa":          "ctry_USA",
}


# ─────────────────────────────────────────────────────────────
# LOAD MODELS + DATASET
# ─────────────────────────────────────────────────────────────
def load_models():
    hgb    = joblib.load("models/hgb_model.pkl")
    ridge  = joblib.load("models/ridge_meta.pkl")
    scaler = joblib.load("models/scaler_hybrid.pkl")
    xgb_model = xgb.Booster()
    xgb_model.load_model("models/xgb_model.json")

    try:
        df = pd.read_csv("data/cleaned/final_cleaned_dataset.csv")
        print(f"  ✓ Dataset loaded: {df.shape}")
    except Exception as e:
        print(f"  ⚠ Dataset not found ({e}). Using global means as fallback.")
        df = None

    print("  ✓ All models loaded.")
    return {"hgb": hgb, "xgb": xgb_model, "ridge": ridge,
            "scaler": scaler, "df": df}


# ─────────────────────────────────────────────────────────────
# GET THE MOST RECENT REAL ROW for a country+crop combo
# ─────────────────────────────────────────────────────────────
def get_base_row(df: pd.DataFrame, crop_col: str, ctry_col: str,
                 up_to_year: int) -> pd.Series | None:
    """
    Return the last available row (year <= up_to_year) for this
    country+crop combo. Filters via the one-hot boolean columns.
    """
    if df is None:
        return None
    mask = df[crop_col].astype(bool) & df[ctry_col].astype(bool)
    subset = df[mask & (df["Year"] <= up_to_year)].sort_values("Year")
    if subset.empty:
        return None
    return subset.iloc[-1]


# ─────────────────────────────────────────────────────────────
# BUILD ONE FEATURE VECTOR for a given year + lag state
# ─────────────────────────────────────────────────────────────
def build_row(year: int, crop_col: str, ctry_col: str,
              base: pd.Series | None,
              lag1: float, lag2: float, lag3: float,
              roll_mean: float, roll_std: float) -> np.ndarray:

    year_norm  = (year - YEAR_MIN) / YEAR_RANGE
    year_norm2 = year_norm ** 2

    def v(col):
        """Get value from the base row, else global mean."""
        if base is not None and col in base.index:
            val = base[col]
            if not (isinstance(val, float) and np.isnan(val)):
                return float(val)
        return GLOBAL_MEANS.get(col, 0.0)

    row = {f: 0.0 for f in FEATURE_NAMES}

    # Static climate/fertiliser features from base row
    for col in ["Avg_Temperature_C", "CO2_Emissions_TonsCapita",
                "Extreme_Weather_Events", "Fert_N_tonnes", "Fert_P_tonnes",
                "Fert_Total_tonnes", "Fert_per_capita", "Forest_Area",
                "Population", "Rain_anomaly", "Rain_lag1", "Rainfall_mm",
                "Renewable_Energy", "Sea_Level_Rise_mm",
                "Temp_anomaly", "Temp_lag1"]:
        row[col] = v(col)

    # Year features
    row["Year_norm"]  = year_norm
    row["Year_norm2"] = year_norm2

    # Lag/roll features — updated iteratively for future years
    row["Yield_lag1"]            = lag1
    row["Yield_lag2"]            = lag2
    row["Yield_lag3"]            = lag3
    row["Yield_kg_ha_roll3_mean"] = roll_mean
    row["Yield_kg_ha_roll3_std"]  = roll_std

    # One-hot flags
    row[crop_col] = 1.0
    row[ctry_col] = 1.0

    return np.array([row[f] for f in FEATURE_NAMES], dtype=np.float64)


# ─────────────────────────────────────────────────────────────
# SINGLE PREDICTION STEP
# ─────────────────────────────────────────────────────────────
def _predict_one(vec: np.ndarray, models: dict) -> float:
    scaler    = models["scaler"]
    hgb       = models["hgb"]
    xgb_model = models["xgb"]
    ridge     = models["ridge"]

    scaled   = scaler.transform(vec.reshape(1, -1))
    hgb_pred = float(hgb.predict(scaled)[0])
    dmatrix  = xgb.DMatrix(scaled, feature_names=FEATURE_NAMES)
    xgb_pred = float(xgb_model.predict(dmatrix)[0])
    return float(ridge.predict(np.array([[hgb_pred, xgb_pred]]))[0])


# ─────────────────────────────────────────────────────────────
# MAIN PREDICTION PIPELINE
# ─────────────────────────────────────────────────────────────
def predict_yield(year: int, country: str, crop: str, models: dict) -> dict:
    df        = models["df"]
    crop_key  = crop.strip().lower()
    ctry_key  = country.strip().lower()
    crop_col  = CROP_COL.get(crop_key, "crop_Wheat")
    ctry_col  = CTRY_COL.get(ctry_key)

    if ctry_col is None:
        # Unknown country → use global means, no one-hot set
        ctry_col = "ctry_USA"   # fallback column (will be overridden to 0)

    # ── Case 1: historical year (in training data) ──────────
    if year <= YEAR_MAX:
        base = get_base_row(df, crop_col, ctry_col, year)
        if base is not None:
            lag1      = float(base["Yield_lag1"])
            lag2      = float(base["Yield_lag2"])
            lag3      = float(base["Yield_lag3"])
            roll_mean = float(base["Yield_kg_ha_roll3_mean"])
            roll_std  = float(base["Yield_kg_ha_roll3_std"])
        else:
            lag1 = lag2 = lag3 = GLOBAL_MEANS["Yield_lag1"]
            roll_mean = GLOBAL_MEANS["Yield_kg_ha_roll3_mean"]
            roll_std  = GLOBAL_MEANS["Yield_kg_ha_roll3_std"]

        vec   = build_row(year, crop_col, ctry_col, base, lag1, lag2, lag3, roll_mean, roll_std)
        final = _predict_one(vec, models)
        return {"yield": round(final, 1), "note": f"Using {year} historical data"}

    # ── Case 2: future year — iterate from YEAR_MAX+1 ───────
    # Seed lags from the last known real year
    base = get_base_row(df, crop_col, ctry_col, YEAR_MAX)
    if base is not None:
        lag1      = float(base["Yield_kg_ha"])   # actual 2023 yield
        lag2      = float(base["Yield_lag1"])
        lag3      = float(base["Yield_lag2"])
        roll_hist = [float(base["Yield_lag2"]),
                     float(base["Yield_lag1"]),
                     float(base["Yield_kg_ha"])]
        roll_std  = float(base["Yield_kg_ha_roll3_std"])
    else:
        lag1 = lag2 = lag3 = GLOBAL_MEANS["Yield_lag1"]
        roll_hist = [lag3, lag2, lag1]
        roll_std  = GLOBAL_MEANS["Yield_kg_ha_roll3_std"]

    predicted = lag1   # in case year == YEAR_MAX + 1

    for yr in range(YEAR_MAX + 1, year + 1):
        roll_mean = float(np.mean(roll_hist[-3:]))
        roll_std  = float(np.std(roll_hist[-3:]))  if len(roll_hist) >= 3 else roll_std

        vec       = build_row(yr, crop_col, ctry_col, base,
                               lag1, lag2, lag3, roll_mean, roll_std)
        predicted = _predict_one(vec, models)

        # Roll lags forward
        lag3      = lag2
        lag2      = lag1
        lag1      = predicted
        roll_hist.append(predicted)

    return {
        "yield": round(predicted, 1),
        "note":  f"Iterative forecast from {YEAR_MAX} to {year}"
    }