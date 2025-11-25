from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'cse6748-construction-cost-estimator'

# Project information
PROJECT_INFO = {
    'course': 'CSE6748 - Applied Analytics Practicum',
    'institution': 'Georgia Institute of Technology',
    'semester': 'Fall 2025',
    'client': 'Construction Cost Database LLC',
    'target_mape': '< 25%'
}

# Global variables
model = None
df = None
categorical_values = {}
numerical_ranges = {}
feature_columns = []
model_metrics = {}
dataset_stats = {}

def calculate_dataset_stats():
    """Calculate comprehensive dataset statistics"""
    global df, dataset_stats
    
    if df is None:
        return {}
    
    target = 'total_project_cost_normalized_2025'
    
    try:
        stats = {
            'total_projects': int(len(df)),
            'total_projects_formatted': f"{len(df):,}",
            'avg_cost': float(df[target].mean()),
            'avg_cost_formatted': f"${df[target].mean():,.2f}",
            'median_cost': float(df[target].median()),
            'median_cost_formatted': f"${df[target].median():,.2f}",
            'min_cost': float(df[target].min()),
            'min_cost_formatted': f"${df[target].min():,.2f}",
            'max_cost': float(df[target].max()),
            'max_cost_formatted': f"${df[target].max():,.2f}",
            'std_cost': float(df[target].std()),
            'std_cost_formatted': f"${df[target].std():,.2f}",
            'total_cost': float(df[target].sum()),
            'total_cost_formatted': f"${df[target].sum():,.2f}"
        }
        
        # Count by project type
        if 'project_type' in df.columns:
            stats['project_types'] = int(df['project_type'].nunique())
            stats['top_project_type'] = str(df['project_type'].mode()[0]) if len(df['project_type'].mode()) > 0 else 'N/A'
        
        # Count by state
        if 'project_state' in df.columns:
            stats['states'] = int(df['project_state'].nunique())
            stats['top_state'] = str(df['project_state'].mode()[0]) if len(df['project_state'].mode()) > 0 else 'N/A'
        
        # Count by budget range
        if 'official_budget_range' in df.columns:
            stats['budget_ranges'] = int(df['official_budget_range'].nunique())
        
        # Year range if available
        if 'inflation_factor' in df.columns:
            stats['avg_inflation_factor'] = float(df['inflation_factor'].mean())
        
        dataset_stats = stats
        return stats
    
    except Exception as e:
        print(f"Error calculating stats: {str(e)}")
        return {}

# Context processor - automatically passes variables to all templates
@app.context_processor
def inject_global_vars():
    """Inject project info, stats, and metrics into all templates"""
    return {
        'project': PROJECT_INFO,
        'stats': dataset_stats if dataset_stats else {},
        'metrics': model_metrics if model_metrics else {},
        'has_data': df is not None,
        'has_model': model is not None
    }

def load_data_and_model():
    """Load the dataset and trained model"""
    global model, df, categorical_values, numerical_ranges, feature_columns, model_metrics
    
    try:
        # Load dataset
        csv_path = 'data/base_data_for_model.csv'
        if not os.path.exists(csv_path):
            print(f"❌ Error: {csv_path} not found.")
            return False
        
        df = pd.read_csv(csv_path, low_memory=False)
        print(f"✅ Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Calculate dataset statistics
        calculate_dataset_stats()
        print(f"✅ Dataset statistics calculated")
        
        # Define target
        target = 'total_project_cost_normalized_2025'
        
        if target not in df.columns:
            print(f"❌ Error: Target column '{target}' not found")
            return False
        
        # EXACT FEATURES FROM GITHUB (matching working version)
        potential_features = [
            'inflation_factor',
            'official_budget_range',
            'ciqs_complexity_category',
            'cnt_division',
            'cnt_item_code',
            'county_name',
            'area_type',
            'acf',
            'project_type',
            'project_category',
            'project_state'
        ]
        
        # Filter only columns that exist in the dataset
        feature_columns = [col for col in potential_features if col in df.columns]
        
        print(f"✅ Using {len(feature_columns)} features: {feature_columns}")
        
        # Store categorical values for dropdowns
        for col in feature_columns:
            if df[col].dtype == 'object':
                unique_vals = df[col].dropna().unique().tolist()
                categorical_values[col] = sorted([str(v) for v in unique_vals])
        
        print(f"✅ Loaded {len(categorical_values)} categorical features")
        
        # Store numerical ranges
        for col in feature_columns:
            if df[col].dtype in ['int64', 'float64']:
                numerical_ranges[col] = {
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'median': float(df[col].median()),
                    'std': float(df[col].std())
                }
        
        print(f"✅ Loaded {len(numerical_ranges)} numerical features")
        
        # Load or train model
        model_path = 'models/construction_cost_model.pkl'
        os.makedirs('models', exist_ok=True)
        
        should_retrain = False
        
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                print("✅ Model loaded successfully")
                
                metrics_path = 'models/model_metrics.json'
                if os.path.exists(metrics_path):
                    import json
                    with open(metrics_path, 'r') as f:
                        model_metrics = json.load(f)
                        print(f"✅ Model metrics loaded: R²={model_metrics.get('r2_score', 0):.4f}, MAE=${model_metrics.get('mae', 0):,.2f}")
            except Exception as e:
                print(f"⚠️ Could not load model: {str(e)}")
                should_retrain = True
        else:
            should_retrain = True
        
        if should_retrain:
            print("🔄 Training new model...")
            model, model_metrics = train_new_model(df, feature_columns, target)
            joblib.dump(model, model_path)
            
            import json
            with open('models/model_metrics.json', 'w') as f:
                json.dump(model_metrics, f, indent=2)
            
            print("✅ Model trained and saved")
        
        return True
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def train_new_model(df, feature_columns, target):
    """Train a new model - EXACT GITHUB VERSION"""
    
    # Prepare data - include target in columns
    available_columns = feature_columns + [target]
    df_clean = df[available_columns].copy()
    
    # Handle missing values
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if df_clean[col].dtype in ['int64', 'float64']:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            else:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
    
    # Prepare features and target
    features = [col for col in df_clean.columns if col != target]
    X = df_clean[features]
    y = df_clean[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Identify categorical and numerical columns
    categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
    numerical_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    # Create pipeline with Random Forest - EXACT SAME AS YESTERDAY
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Train model
    print("🚀 Training Random Forest model...")
    pipeline.fit(X_train, y_train)
    print("✅ Training completed")
    
    # Evaluate (optional - for metrics)
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
    y_pred_test = pipeline.predict(X_test)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
    
    print(f"\n📈 Model Performance:")
    print(f"  R² Score: {test_r2:.4f}")
    print(f"  MAE: ${test_mae:,.2f}")
    print(f"  RMSE: ${test_rmse:,.2f}")
    print(f"  MAPE: {mape:.2f}%\n")
    
    from datetime import datetime
    metrics = {
        'r2_score': float(test_r2),
        'mae': float(test_mae),
        'mae_formatted': f"${test_mae:,.2f}",
        'rmse': float(test_rmse),
        'rmse_formatted': f"${test_rmse:,.2f}",
        'mape': float(mape),
        'mape_formatted': f"{mape:.2f}%",
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'features': feature_columns,
        'n_features': len(feature_columns),
        'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return pipeline, metrics
def get_similar_projects(input_data):
    """Find similar projects in the dataset"""
    global df
    
    if df is None:
        return {
            'count': 0,
            'avg_cost_formatted': 'N/A',
            'median_cost_formatted': 'N/A',
            'min_cost_formatted': 'N/A',
            'max_cost_formatted': 'N/A',
            'std_cost_formatted': 'N/A',
            'match_type': 'none'
        }
    
    target = 'total_project_cost_normalized_2025'
    filtered_df = df.copy()
    
    # Filter by key categorical fields
    filter_fields = ['project_type', 'official_budget_range', 'ciqs_complexity_category']
    
    filters_applied = []
    for field in filter_fields:
        if field in input_data and field in df.columns:
            try:
                filtered_df = filtered_df[filtered_df[field] == input_data[field]]
                filters_applied.append(field)
            except Exception as e:
                print(f"  ⚠ Error filtering by {field}: {e}")
    
    print(f"  Filters applied: {filters_applied}")
    print(f"  Similar projects found: {len(filtered_df)}")
    
    # Return statistics based on how many similar projects we found
    if len(filtered_df) >= 5:
        # Exact match - enough similar projects
        return {
            'count': len(filtered_df),
            'avg_cost': float(filtered_df[target].mean()),
            'avg_cost_formatted': f"${filtered_df[target].mean():,.2f}",
            'median_cost': float(filtered_df[target].median()),
            'median_cost_formatted': f"${filtered_df[target].median():,.2f}",
            'min_cost': float(filtered_df[target].min()),
            'min_cost_formatted': f"${filtered_df[target].min():,.2f}",
            'max_cost': float(filtered_df[target].max()),
            'max_cost_formatted': f"${filtered_df[target].max():,.2f}",
            'std_cost': float(filtered_df[target].std()),
            'std_cost_formatted': f"${filtered_df[target].std():,.2f}",
            'match_type': 'exact'
        }
    elif len(filtered_df) > 0:
        # Partial match - some similar projects
        return {
            'count': len(filtered_df),
            'avg_cost': float(filtered_df[target].mean()),
            'avg_cost_formatted': f"${filtered_df[target].mean():,.2f}",
            'median_cost': float(filtered_df[target].median()),
            'median_cost_formatted': f"${filtered_df[target].median():,.2f}",
            'min_cost': float(filtered_df[target].min()),
            'min_cost_formatted': f"${filtered_df[target].min():,.2f}",
            'max_cost': float(filtered_df[target].max()),
            'max_cost_formatted': f"${filtered_df[target].max():,.2f}",
            'std_cost': float(filtered_df[target].std()),
            'std_cost_formatted': f"${filtered_df[target].std():,.2f}",
            'match_type': 'partial'
        }
    else:
        # No matches - use overall statistics
        return {
            'count': len(df),
            'avg_cost': float(df[target].mean()),
            'avg_cost_formatted': f"${df[target].mean():,.2f}",
            'median_cost': float(df[target].median()),
            'median_cost_formatted': f"${df[target].median():,.2f}",
            'min_cost': float(df[target].min()),
            'min_cost_formatted': f"${df[target].min():,.2f}",
            'max_cost': float(df[target].max()),
            'max_cost_formatted': f"${df[target].max():,.2f}",
            'std_cost': float(df[target].std()),
            'std_cost_formatted': f"${df[target].std():,.2f}",
            'match_type': 'overall'
        }
# ============= ROUTES =============


@app.route('/')
def home():
    """Home page"""
    return render_template('home.html')

@app.route('/eda')
def eda():
    """EDA page with dataset statistics"""
    return render_template('eda.html')

@app.route('/model_comparison')
def model_comparison():
    """Model comparison page"""
    return render_template('model_comparison.html')

@app.route('/dashboard')
def dashboard():
    """Performance dashboard page"""
    return render_template('dashboard.html')

@app.route('/cost_estimator')  # ← Changed from /cost_estimation
def cost_estimator():           # ← Changed function name
    """Cost estimator form page"""
    if df is None or model is None:
        return render_template('error.html', 
                             error='System not ready',
                             message='Dataset or model not loaded. Please restart the application.')
    
    return render_template('cost_estimator.html',  # ← Make sure template name matches
                         categorical_values=categorical_values,
                         numerical_ranges=numerical_ranges,
                         feature_columns=feature_columns)

@app.route('/estimate_cost', methods=['POST'])
def estimate_cost():
    """Handle cost estimation prediction"""
    try:
        print("\n" + "="*60)
        print("📝 ESTIMATION REQUEST RECEIVED")
        print("="*60)
        
        if model is None:
            print("❌ ERROR: Model not loaded")
            return jsonify({'success': False, 'error': 'Model not loaded'}), 500
        
        if not feature_columns:
            print("❌ ERROR: Feature columns not defined")
            return jsonify({'success': False, 'error': 'Feature columns not configured'}), 500
        
        # Get form data
        input_data = {}
        
        print("\n📋 Form Data Received:")
        print(f"Request form keys: {list(request.form.keys())}")
        
        for col in feature_columns:
            value = request.form.get(col)
            if value:
                if col in numerical_ranges:
                    try:
                        input_data[col] = float(value)
                        print(f"  ✓ {col}: {value} (numerical)")
                    except ValueError as e:
                        input_data[col] = numerical_ranges[col]['median']
                        print(f"  ⚠ {col}: using median (invalid input: {e})")
                else:
                    input_data[col] = value
                    print(f"  ✓ {col}: {value} (categorical)")
            else:
                print(f"  ⚠ {col}: missing from form")
        
        if not input_data:
            print("❌ ERROR: No input data received")
            return jsonify({'success': False, 'error': 'No input data provided. Please fill all fields.'}), 400
        
        print(f"\n✅ Collected {len(input_data)}/{len(feature_columns)} features")
        
        # Create DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Fill missing columns with defaults
        for col in feature_columns:
            if col not in input_df.columns:
                if col in numerical_ranges:
                    input_df[col] = numerical_ranges[col]['median']
                    print(f"  + Added {col} = {numerical_ranges[col]['median']} (median)")
                elif col in categorical_values and categorical_values[col]:
                    input_df[col] = categorical_values[col][0]
                    print(f"  + Added {col} = {categorical_values[col][0]} (first value)")
                else:
                    print(f"  ⚠ Cannot fill {col} - no default available")
        
        # Reorder columns to match training
        try:
            input_df = input_df[feature_columns]
        except KeyError as e:
            print(f"❌ ERROR: Missing column: {e}")
            return jsonify({'success': False, 'error': f'Missing required field: {e}'}), 400
        
        print("\n🔮 Making prediction...")
        print(f"Input DataFrame shape: {input_df.shape}")
        print(f"Input DataFrame columns: {list(input_df.columns)}")
        
        prediction = model.predict(input_df)[0]
        print(f"✅ PREDICTION: ${prediction:,.2f}")
        
        # Calculate confidence interval
        lower_bound = prediction * 0.80
        upper_bound = prediction * 1.20
        
        # Get similar projects
        similar_stats = get_similar_projects(input_data)
        
        from datetime import datetime
        result = {
            'success': True,
            'estimated_cost': float(prediction),
            'estimated_cost_formatted': f"${prediction:,.2f}",
            'confidence_interval': {
                'lower': float(lower_bound),
                'upper': float(upper_bound),
                'lower_formatted': f"${lower_bound:,.2f}",
                'upper_formatted': f"${upper_bound:,.2f}"
            },
            'input_data': input_data,
            'similar_projects': similar_stats,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print(f"✅ Returning successful result")
        print("="*60 + "\n")
        
        return jsonify(result)
    
    except Exception as e:
        print(f"\n❌ EXCEPTION in estimate_cost:")
        print(f"  Type: {type(e).__name__}")
        print(f"  Message: {str(e)}")
        import traceback
        traceback.print_exc()
        print("="*60 + "\n")
        
        # Return detailed error in development
        return jsonify({
            'success': False, 
            'error': f'{type(e).__name__}: {str(e)}',
            'details': traceback.format_exc()
        }), 500

@app.route('/data_overview')
def data_overview():
    """Data overview page"""
    return render_template('data_overview.html')

@app.route('/documentation')
def documentation():
    """Documentation page"""
    return render_template('documentation.html')
# Debug: Print all registered routes
def list_routes():
    print("\n📋 Registered Routes:")
    for rule in app.url_map.iter_rules():
        methods = ','.join(rule.methods)
        print(f"  {rule.endpoint:30s} {methods:20s} {rule}")
    print()
    
    
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🏗️  Construction Cost Estimator")
    print("="*60 + "\n")
    
    if load_data_and_model():
        list_routes()  # Add this line
        
        print("\n" + "="*60)
        print("✅ System Ready!")
        print("="*60)
        print("🌐 Access at: http://localhost:5000")
        print("📊 Dataset: Loaded")
        print("🤖 Model: Ready")
        print("="*60 + "\n")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n" + "="*60)
        print("❌ Failed to initialize")
        print("="*60)
        print("Please ensure 'data/base_data_for_model.csv' exists")
        print("="*60 + "\n")