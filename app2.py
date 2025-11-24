import os
import socket
import locale
import joblib
import numpy as np
import pandas as pd

from flask import (
    Flask,
    render_template,
    request,
    jsonify
)

from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
except ImportError:
    # Fallback if colorama is not installed
    class Dummy:
        def __getattr__(self, name):
            return ""
    Fore = Style = Dummy()


# -------------------------------------------------------------------
# Basic Flask + locale setup
# -------------------------------------------------------------------
locale.setlocale(locale.LC_ALL, "")

app = Flask(__name__)


# Template filter to format numbers nicely
@app.template_filter("format_number")
def format_number(value):
    try:
        return f"{float(value):,.2f}"
    except Exception:
        return value


# -------------------------------------------------------------------
# Paths and global variables
# -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "construction_cost_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "model_metrics.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "base_data_for_model.csv")

os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

PROJECT_INFO = {
    "course": "CSE6748 - Applied Analytics Practicum",
    "institution": "Georgia Institute of Technology",
    "semester": "Fall 2025",
    "client": "Construction Cost Database LLC",
    "target_mape": "< 25%"
}


# Global variables
model = None
model_metrics = {}
df = None


# Mapping from project_type -> typical project_category
PROJECT_TYPE_TO_CATEGORY = {}

# Valid values for categorical features
STATE_FACTORS = {}
TYPE_FACTORS = {}
CATEGORY_FACTORS = {}


DIV_RANGE = (1, 50)
ITEM_RANGE = (1, 500)
CSI_RANGE = (1, 150)
ACF_RANGE = (0.5, 2.0)
CPI_RANGE = (200, 350)
INFLATION_RANGE = (0.8, 1.5)


# -------------------------------------------------------------------
# Helper: default row based on similar projects
# -------------------------------------------------------------------
def get_default_row(filters):
    """
    Given a set of high-level filters, return typical (median/mode) values
    for the technical features needed by the model.

    filters: dict like {
        "project_state": "OH",
        "project_type": "School",
        "project_category": "New Construction",
        "area_type": "Urban",
        "county_name": "Franklin County"
    }
    """
    global df

    subset = df.copy()

    # Apply filters step by step so it degrades gracefully
    for col, val in filters.items():
        if val and col in subset.columns:
            subset = subset[subset[col] == val]

    # If filters shrink too much, back off to full dataset
    if len(subset) < 50:
        subset = df

    def med(col, fallback=None):
        if col not in subset.columns:
            return fallback
        return subset[col].median()

    def mode(col, fallback=None):
        if col not in subset.columns:
            return fallback
        vals = subset[col].dropna()
        if vals.empty:
            return fallback
        return vals.mode().iloc[0]

    defaults = {}

    # Technical counts
    defaults["cnt_division"] = int(med("cnt_division", 10))
    defaults["cnt_item_code"] = int(med("cnt_item_code", 30))
    defaults["cnt_csi_grp_unq"] = int(med("cnt_csi_grp_unq", 15))

    # Cost factors
    defaults["acf"] = float(med("acf", 1.0))
    defaults["cpi"] = float(med("cpi", 240.0))
    defaults["inflation_factor"] = float(med("inflation_factor", 1.0))

    # Categories that the model likes
    defaults["official_budget_range"] = mode("official_budget_range", "Unknown")
    defaults["construction_category"] = mode("construction_category", "Unknown")
    # Area type that usually goes with this combination
    defaults["area_type"] = mode("area_type", None)

    # Coordinates (if present)
    defaults["project_latitude"] = float(med("project_latitude", 0.0))
    defaults["project_longitude"] = float(med("project_longitude", 0.0))

    return defaults


# -------------------------------------------------------------------
# Fallback model if trained model is not found
# -------------------------------------------------------------------
def create_dynamic_fallback_model():
    """
    Create a simple fallback model using the available dataset.

    This is only used when the trained model .pkl cannot be loaded.
    It fits a small RandomForest on numeric features and returns
    both the model and reasonable metrics.
    """
    global df  # use already loaded dataset

    # ---- Case 1: no data at all ----
    if df is None or len(df) == 0:
        class MedianModel:
            def __init__(self, value):
                self.value = value

            def predict(self, X):
                return np.full(shape=(len(X),), fill_value=self.value)

        median_cost = 1_000_000.0
        print(f"{Fore.YELLOW}No data. Using dummy median model {median_cost}{Style.RESET_ALL}")
        return MedianModel(median_cost), {
            "model_type": "Median-based Fallback",
            "mape": 25.0,
            "rmse": 300_000.0,
            "r2": 0.0,
        }

    target_col = "total_project_cost_normalized_2025"
    if target_col not in df.columns:
        # ---- Case 2: target column missing ----
        class MedianModel:
            def __init__(self, value):
                self.value = value

            def predict(self, X):
                return np.full(shape=(len(X),), fill_value=self.value)

        median_cost = 1_000_000.0
        print(f"{Fore.YELLOW}Target column missing. Using dummy median model{Style.RESET_ALL}")
        return MedianModel(median_cost), {
            "model_type": "Median-based Fallback",
            "mape": 25.0,
            "rmse": 300_000.0,
            "r2": 0.0,
        }

    # ---- Case 3: we have data + target → train a small RF on numeric features ----
    numeric_cols = [
        "cnt_division",
        "cnt_item_code",
        "cnt_csi_grp_unq",
        "acf",
        "cpi",
        "inflation_factor",
        "complexity_numeric",
        "budget_midpoint",
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    df_model = df.dropna(subset=[target_col] + numeric_cols).copy()
    if len(df_model) < 100:
        # Not enough clean rows → median model again, but based on actual data
        class MedianModel:
            def __init__(self, value):
                self.value = value

            def predict(self, X):
                return np.full(shape=(len(X),), fill_value=self.value)

        median_cost = float(df[target_col].median())
        print(f"{Fore.YELLOW}Insufficient clean rows. Using simple median model{Style.RESET_ALL}")
        return MedianModel(median_cost), {
            "model_type": "Median-based Fallback",
            "mape": 25.0,
            "rmse": float(df[target_col].std()),
            "r2": 0.0,
        }

    X = df_model[numeric_cols]
    y = df_model[target_col].astype(float)

    rf = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X, y)

    # ---- Safe metric calculations ----
    preds = rf.predict(X)

    # percentage error
    errors_pct = np.abs((y - preds) / np.maximum(y, 1.0)) * 100.0
    mape = float(errors_pct.mean())

    rmse = float(np.sqrt(np.mean((y - preds) ** 2)))

    from sklearn.metrics import r2_score
    r2 = float(r2_score(y, preds))

    metrics = {
        "model_type": "RandomForest (Fallback)",
        "mape": round(mape, 2),
        "rmse": rmse,
        "r2": round(r2, 4),
    }

    print(f"{Fore.YELLOW}Using dynamic fallback RandomForest model{Style.RESET_ALL}")
    return rf, metrics


# -------------------------------------------------------------------
# Data/model loading
# -------------------------------------------------------------------
def load_data_and_model():
    """Load dataset, model, metrics, and prepare lookup dicts."""
    global model, model_metrics, df
    global STATE_FACTORS, TYPE_FACTORS, CATEGORY_FACTORS
    global COUNTY_FACTORS, AREA_TYPES, COMPLEXITY_CATEGORIES
    global PROJECT_TYPE_TO_CATEGORY

    try:
        # Load data
        df_ = pd.read_csv(DATA_PATH, low_memory=False)
        print(f"Dataset shape: {df_.shape}")

        # --- Ensure 'region' column, but do NOT overwrite if it already exists ---
        if (
            "region" not in df_.columns
            and "project_state" in df_.columns
            and "project_latitude" in df_.columns
            and "project_longitude" in df_.columns
        ):
            state_coords = {}
            for state in df_["project_state"].dropna().unique():
                state_data = df_[df_["project_state"] == state]
                lat = state_data["project_latitude"].median()
                lng = state_data["project_longitude"].median()
                if not np.isnan(lat) and not np.isnan(lng):
                    state_coords[state] = (lat, lng)

            if state_coords:
                states = list(state_coords.keys())
                coords = np.array([state_coords[s] for s in states])

                optimal_k = min(4, len(coords))
                kmeans = KMeans(n_clusters=optimal_k, random_state=42)
                kmeans.fit(coords)

                state_to_region = {}
                for i, state in enumerate(states):
                    region = kmeans.predict(
                        np.array([state_coords[state]]).reshape(1, -1)
                    )[0]
                    state_to_region[state] = f"Region_{region}"

                df_["region"] = df_["project_state"].map(state_to_region)
                print(f"Created {optimal_k} geographic regions.")


        # Assign globals
        df = df_

        # Fill categorical sets
        if "project_state" in df.columns:
            STATE_FACTORS = {
                s: s for s in sorted(df["project_state"].dropna().unique())
            }
        


        # NEW: build mapping project_type -> most common project_category
        global PROJECT_TYPE_TO_CATEGORY
        if "project_type" in df.columns and "project_category" in df.columns:
            PROJECT_TYPE_TO_CATEGORY = (
                df.dropna(subset=["project_type", "project_category"])
                  .groupby("project_type")["project_category"]
                  .agg(lambda x: x.mode().iloc[0])
                  .to_dict()
            )
            print(f"Built PROJECT_TYPE_TO_CATEGORY for {len(PROJECT_TYPE_TO_CATEGORY)} types")
        else:
            PROJECT_TYPE_TO_CATEGORY = {}
            print("Warning: cannot build type→category mapping (columns missing).")

        if "county_name" in df.columns:
            COUNTY_FACTORS = {
                c: c for c in sorted(df["county_name"].dropna().unique())
            }
        if "area_type" in df.columns:
            AREA_TYPES = {
                a: a for a in sorted(df["area_type"].dropna().unique())
            }
        if "ciqs_complexity_category" in df.columns:
            COMPLEXITY_CATEGORIES = {
                c: c
                for c in sorted(
                    df["ciqs_complexity_category"].dropna().unique()
                )
            }

        # Load model
        model = None
        model_metrics = {}

        if os.path.exists(MODEL_PATH):
            try:
                model = joblib.load(MODEL_PATH)
                print(f"Loaded model from {MODEL_PATH}")
            except Exception as e:
                print(f"Error loading model from {MODEL_PATH}: {e}")
        else:
            print(f"Model file not found at {MODEL_PATH}")

        if os.path.exists(METRICS_PATH):
            try:
                mm = joblib.load(METRICS_PATH)
                if isinstance(mm, dict):
                    model_metrics = mm
                else:
                    print(f"Warning: metrics file {METRICS_PATH} is not a dict; ignoring.")
            except Exception as e:
                print(f"Error loading metrics from {METRICS_PATH}: {e}")
        else:
            print(f"Metrics file not found at {METRICS_PATH}")


    except Exception as e:
        print(f"{Fore.RED}Error loading data/model: {e}{Style.RESET_ALL}")

    # Replace the hardcoded model with the dynamic one if needed
        # Replace with dynamic fallback model if trained model is not available
    if model is None:
        print(
            f"{Fore.YELLOW}WARNING: Using dynamic fallback model based on "
            f"dataset patterns. Real model not found.{Style.RESET_ALL}"
        )
        fb_model, fb_metrics = create_dynamic_fallback_model()
        model = fb_model
        model_metrics = fb_metrics



# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------
@app.route("/")
def home():
    # Basic stats for front page cards
    if df is None or len(df) == 0:
        stats = {
            "total_projects": 0,
            "avg_cost": 0,
            "min_cost": 0,
            "max_cost": 0,
            "states": 0,
            "cities": 0,
        }
    else:
        target = df["total_project_cost_normalized_2025"]
        stats = {
            "total_projects": len(df),
            "avg_cost": f"{target.mean():,.0f}",
            "min_cost": f"{target.min():,.0f}",
            "max_cost": f"{target.max():,.0f}",
            "states": df["project_state"].nunique()
            if "project_state" in df.columns
            else 0,
            "cities": df["project_city"].nunique()
            if "project_city" in df.columns
            else 0,
        }

    return render_template("index.html", stats=stats)


@app.route("/eda")
def eda():
    if df is None or len(df) == 0:
        stats = {
            "total_projects": 0,
            "avg_cost": 0,
            "min_cost": 0,
            "max_cost": 0,
            "states": 0,
            "cities": 0,
        }
    else:
        target = df["total_project_cost_normalized_2025"]
        stats = {
            "total_projects": len(df),
            "avg_cost": f"{target.mean():,.0f}",
            "min_cost": f"{target.min():,.0f}",
            "max_cost": f"{target.max():,.0f}",
            "states": df["project_state"].nunique()
            if "project_state" in df.columns
            else 0,
            "cities": df["project_city"].nunique()
            if "project_city" in df.columns
            else 0,
        }

    return render_template("eda.html", stats=stats)


@app.route("/model_comparison")
def model_comparison():
    # For now just pass model_metrics; templates can show what they need
    comparison = model_metrics.copy()
    return render_template("model_comparison.html", comparison=comparison)


@app.route("/dashboard")
def dashboard():
    # You can extend this as needed; passing df summary is enough for layout
    return render_template("dashboard.html")


@app.route("/data_overview")
def data_overview():
    # For schema/table overview
    columns = []
    if df is not None:
        for col in df.columns:
            columns.append(
                {"name": col, "dtype": str(df[col].dtype)}
            )
    return render_template("data_overview.html", columns=columns)


@app.route('/cost_estimator')
def cost_estimator():
    """Cost estimator form page"""
    global df

    # Make sure df is loaded
    if df is None:
        try:
            df = pd.read_csv(DATA_PATH)
            print(f"[cost_estimator] Loaded df from {DATA_PATH}, shape={df.shape}")
        except Exception as e:
            print(f"[cost_estimator] ERROR loading DATA_PATH: {e}")
            df_local = None
        else:
            df_local = df
    else:
        df_local = df

    def unique_values(col_name):
        if df_local is not None and col_name in df_local.columns:
            return sorted(df_local[col_name].dropna().unique())
        return []

    # Pull dropdown values directly from the dataframe
    states = unique_values("project_state")
    project_types = unique_values("project_type")
    project_categories = unique_values("project_category")
    area_types = unique_values("area_type")
    complexity_categories = unique_values("ciqs_complexity_category")

    # Build mapping: project_type -> MOST COMMON project_category
    type_category_map = {}
    if df_local is not None and "project_type" in df_local.columns and "project_category" in df_local.columns:
        type_category_map = (
            df_local.dropna(subset=["project_type", "project_category"])
                    .groupby("project_type")["project_category"]
                    .agg(lambda s: s.mode().iloc[0])
                    .to_dict()
        )

    return render_template(
        "cost_estimator.html",
        states=states,
        project_types=project_types,
        project_categories=project_categories,
        area_types=area_types,
        complexity_categories=complexity_categories,
        csi_min=int(CSI_RANGE[0]),
        csi_max=int(CSI_RANGE[1]),
        div_min=int(DIV_RANGE[0]),
        div_max=int(DIV_RANGE[1]),
        item_min=int(ITEM_RANGE[0]),
        item_max=int(ITEM_RANGE[1]),
        acf_min=round(ACF_RANGE[0], 2),
        acf_max=round(ACF_RANGE[1], 2),
        cpi_min=round(CPI_RANGE[0], 2),
        cpi_max=round(CPI_RANGE[1], 2),
        inf_min=round(INFLATION_RANGE[0], 2),
        inf_max=round(INFLATION_RANGE[1], 2),
        type_category_map=type_category_map,   # <- important for JS
    )



@app.route("/predict", methods=["POST"])
def predict():
    """Handle form submission and make prediction."""
    official_budget_range_form = request.form.get("official_budget_range", "").strip()

    try:
        project_state = request.form.get("project_state", "")
        project_type = request.form.get("project_type", "")
        project_category_form = request.form.get("project_category", "")
        area_type = request.form.get("area_type", "")
        county_name = request.form.get("county_name", "")
        project_city = request.form.get("project_city", "")
        ciqs_cat = request.form.get("ciqs_complexity_category", "")

        # Resolve project_category from project_type if we have a mapping
        project_category = PROJECT_TYPE_TO_CATEGORY.get(project_type, project_category_form)

        # Use high-level info to pull typical values from the dataset
        defaults = get_default_row({
            "project_state": project_state,
            "project_type": project_type,
            "project_category": project_category,
            "area_type": area_type,
            "county_name": county_name,
        })

        # If user did not choose area type, use typical area_type from the data
        if not area_type:
            area_type = defaults.get("area_type", area_type)

        # Allow user to override defaults if they provided numbers
        def parse_or_default(name, cast, fallback_key):
            raw = request.form.get(name, "").strip()
            if raw == "":
                return defaults[fallback_key]
            try:
                return cast(raw)
            except ValueError:
                return defaults[fallback_key]

        cnt_division = parse_or_default("cnt_division", int, "cnt_division")
        cnt_item_code = parse_or_default("cnt_item_code", int, "cnt_item_code")
        cnt_csi_grp_unq = parse_or_default(
            "cnt_csi_grp_unq", int, "cnt_csi_grp_unq"
        )
        acf = parse_or_default("acf", float, "acf")
        cpi = parse_or_default("cpi", float, "cpi")
        inflation_factor = parse_or_default(
            "inflation_factor", float, "inflation_factor"
        )

        # Categorical values: use user input if provided, otherwise defaults
        official_budget_range = (
            official_budget_range_form or defaults["official_budget_range"]
        )
        construction_category_form = request.form.get("construction_category", "").strip()
        construction_category = (
            construction_category_form or defaults["construction_category"]
        )


        # Region: use region column if present
        if "region" in df.columns:
            region_val = df.loc[
                df["project_state"] == project_state, "region"
            ].mode()
            region = region_val.iloc[0] if not region_val.empty else None
        else:
            region = None

        # Build full input row with everything (for display & engineering)
        input_data = {
            "project_state": project_state,
            "project_type": project_type,
            "project_category": project_category,
            "construction_category": construction_category,
            "project_city": project_city,
            "county_name": county_name,
            "area_type": area_type,
            "official_budget_range": official_budget_range,
            "ciqs_complexity_category": ciqs_cat,
            "cnt_division": cnt_division,
            "cnt_item_code": cnt_item_code,
            "cnt_csi_grp_unq": cnt_csi_grp_unq,
            "acf": acf,
            "cpi": cpi,
            "inflation_factor": inflation_factor,
        }

        # Coordinates if present
        if "project_latitude" in df.columns:
            input_data["project_latitude"] = defaults["project_latitude"]
        if "project_longitude" in df.columns:
            input_data["project_longitude"] = defaults["project_longitude"]

        if region is not None:
            input_data["region"] = region

        # Engineered features (mostly for display; model might or might not use them)
        cnt_csi_safe = max(1, cnt_csi_grp_unq)
        input_data["complexity_score"] = cnt_division * cnt_item_code / cnt_csi_safe
        input_data["economic_factor"] = cpi * inflation_factor

        # === STRICT FEATURE LIST FOR THE MODEL ===
        MODEL_FEATURES = [
            "inflation_factor",
            "official_budget_range",
            "ciqs_complexity_category",
            "cnt_division",
            "cnt_item_code",
            "county_name",
            "area_type",
            "acf",
            "project_type",
            "project_category",
            "project_state",
            "region",
        ]

        X_input = pd.DataFrame([ {f: input_data.get(f) for f in MODEL_FEATURES} ])

        # Use trained model
        prediction = float(model.predict(X_input)[0])



        # Use trained model if available, otherwise fallback
        if model is None:
            # Fallback if model not loaded
            target = df['total_project_cost_normalized_2025'] if df is not None else None
            prediction = float(target.median()) if target is not None else 0.0
            model_type = "Median-based Fallback"
            mape = 25.0
            rmse = float(target.std()) if target is not None else 0.0
            r2 = 0.0
        else:
            prediction = float(model.predict(df_input)[0])

            # Use safe metrics dict
            metrics = model_metrics if isinstance(model_metrics, dict) else {}
            model_type = metrics.get('model_type', 'Trained Model')
            mape = float(metrics.get('mape', 25.0))
            rmse = float(metrics.get('rmse', 300000.0))
            r2 = float(metrics.get('r2', 0.0))


        # Confidence interval based on MAPE
        lower_bound = prediction * (1 - mape / 100.0)
        upper_bound = prediction * (1 + mape / 100.0)

        return render_template(
            "prediction.html",
            prediction=prediction,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            model_metrics={
                "model_type": model_type,
                "mape": mape,
                "rmse": rmse,
                "r2": r2,
            },
            input_data=input_data,
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        return render_template(
            "error.html", message=f"Error making prediction: {str(e)}"
        )


@app.route("/api/estimate", methods=["POST"])
def api_estimate():
    """JSON API for programmatic cost estimation."""
    try:
        data = request.get_json(force=True)

        project_state = data["project_state"]
        project_type = data["project_type"]
        project_category_raw = data.get(
            "project_category",
            list(CATEGORY_FACTORS.keys())[0] if CATEGORY_FACTORS else ""
        )
        area_type = data.get("area_type", "")
        county_name = data.get("county_name", "")
        project_city = data.get("project_city", "")
        ciqs_cat = data.get("ciqs_complexity_category", "")

        # Resolve category from type if we can
        project_category = PROJECT_TYPE_TO_CATEGORY.get(project_type, project_category_raw)

        defaults = get_default_row({
            "project_state": project_state,
            "project_type": project_type,
            "project_category": project_category,
            "area_type": area_type,
            "county_name": county_name,
        })


        if not area_type:
            area_type = defaults.get("area_type", area_type)

        def parse_or_default_json(key, cast, fallback_key):
            raw = data.get(key, None)
            if raw is None:
                return defaults[fallback_key]
            try:
                return cast(raw)
            except (ValueError, TypeError):
                return defaults[fallback_key]

        cnt_division = parse_or_default_json(
            "cnt_division", int, "cnt_division"
        )
        cnt_item_code = parse_or_default_json(
            "cnt_item_code", int, "cnt_item_code"
        )
        cnt_csi_grp_unq = parse_or_default_json(
            "cnt_csi_grp_unq", int, "cnt_csi_grp_unq"
        )
        acf = parse_or_default_json("acf", float, "acf")
        cpi = parse_or_default_json("cpi", float, "cpi")
        inflation_factor = parse_or_default_json(
            "inflation_factor", float, "inflation_factor"
        )

        official_budget_range = defaults["official_budget_range"]
        construction_category = defaults["construction_category"]

        if "region" in df.columns:
            region_val = df.loc[
                df["project_state"] == project_state, "region"
            ].mode()
            region = region_val.iloc[0] if not region_val.empty else None
        else:
            region = None

        input_data = {
            "project_state": project_state,
            "project_type": project_type,
            "project_category": project_category,
            "construction_category": construction_category,
            "project_city": project_city,
            "county_name": county_name,
            "area_type": area_type,
            "official_budget_range": official_budget_range,
            "ciqs_complexity_category": ciqs_cat,
            "cnt_division": cnt_division,
            "cnt_item_code": cnt_item_code,
            "cnt_csi_grp_unq": cnt_csi_grp_unq,
            "acf": acf,
            "cpi": cpi,
            "inflation_factor": inflation_factor,
        }

        if "project_latitude" in df.columns:
            input_data["project_latitude"] = defaults["project_latitude"]
        if "project_longitude" in df.columns:
            input_data["project_longitude"] = defaults["project_longitude"]

        if region is not None:
            input_data["region"] = region

        cnt_csi_safe = max(1, cnt_csi_grp_unq)
        input_data["complexity_score"] = (
            cnt_division * cnt_item_code / cnt_csi_safe
        )
        input_data["economic_factor"] = cpi * inflation_factor

        df_input = pd.DataFrame([input_data])

        if model is None:
            prediction = df["total_project_cost_normalized_2025"].median()
            model_type = "Median-based Fallback"
            mape = 25.0
            rmse = float(df["total_project_cost_normalized_2025"].std())
            r2 = 0.0
        else:
            prediction = float(model.predict(df_input)[0])
            model_type = model_metrics.get("model_type", "Unknown Model")
            mape = model_metrics.get("mape", 25.0)
            rmse = model_metrics.get("rmse", 300_000.0)
            r2 = model_metrics.get("r2", 0.85)

        lower_bound = prediction * (1 - mape / 100.0)
        upper_bound = prediction * (1 + mape / 100.0)

        return jsonify(
            {
                "estimated_cost": prediction,
                "confidence_interval": {
                    "low": lower_bound,
                    "high": upper_bound,
                },
                "model_metrics": {
                    "model_type": model_type,
                    "mape": mape,
                    "rmse": rmse,
                    "r2": r2,
                },
            }
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.route("/documentation")
def documentation():
    metrics = {
        "mape": model_metrics.get("mape", 0),
        "rmse": model_metrics.get("rmse", 0),
        "r2": model_metrics.get("r2", 0),
    }

    team = [
        {
            "name": "Krishna Aryal",
            "role": "Data Scientist / Developer",
            "description": "Georgia Institute of Technology",
        },
        {
            "name": "Kumar Sawan",
            "role": "Data Engineer",
            "description": "Georgia Institute of Technology",
        },
        {
            "name": "Neema Kafwimi",
            "role": "Business Analyst",
            "description": "Georgia Institute of Technology",
        },
    ]

    reports_dir = os.path.join(BASE_DIR, "static", "reports")
    report_files = os.listdir(reports_dir) if os.path.isdir(reports_dir) else []

    return render_template(
        "documentation.html",
        metrics=metrics,
        team=team,
        report_files=report_files,
    )


@app.route("/debug/factors")
def debug_factors():
    """Debug route to inspect categorical factor dictionaries."""
    return jsonify(
        {
            "states": list(STATE_FACTORS.keys()),
            "project_types": list(TYPE_FACTORS.keys()),
            "project_categories": list(CATEGORY_FACTORS.keys()),
            "counties": list(COUNTY_FACTORS.keys()),
            "area_types": list(AREA_TYPES.keys()),
            "complexity_categories": list(COMPLEXITY_CATEGORIES.keys()),
        }
    )

@app.context_processor
def inject_project():
    # makes {{ project }} available in all templates (base.html, etc.)
    return dict(project=PROJECT_INFO)


# -------------------------------------------------------------------
# Run the app
# -------------------------------------------------------------------
if __name__ == '__main__':
    import os
    import socket
    from colorama import init, Fore, Style
    
    init()

    # 🔹 Make sure data + factors are loaded before starting server
    load_data_and_model()

    print("Starting Construction Cost Estimator...")
    print(f"STATE_FACTORS contains {len(STATE_FACTORS)} states")


    # Get the local IP address
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    # Display app information with color
    print("\n" + "=" * 60)
    print(f"{Fore.GREEN}Construction Cost Estimator{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}CSE6748 - Applied Analytics Practicum{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Georgia Institute of Technology{Style.RESET_ALL}")
    print("=" * 60)

    # Start the Flask app
    port = int(os.environ.get('PORT', 5000))

    print(f"\n{Fore.GREEN}Server starting...{Style.RESET_ALL}")
    print("Access the application at:")
    print(f"{Fore.BLUE}http://localhost:{port}/{Style.RESET_ALL}")
    print(f"{Fore.BLUE}http://{local_ip}:{port}/{Style.RESET_ALL}")
    print("\n" + "=" * 60)

    print(f"{Fore.YELLOW}Useful URLs:{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Main page:        {Style.RESET_ALL}http://localhost:{port}/")
    print(f"{Fore.CYAN}Cost Estimator:   {Style.RESET_ALL}http://localhost:{port}/cost_estimator")
    print(f"{Fore.CYAN}API Documentation:{Style.RESET_ALL}http://localhost:{port}/documentation")
    print(f"{Fore.CYAN}Debug Factors:    {Style.RESET_ALL}http://localhost:{port}/debug/factors")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=port)
