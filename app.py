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
from sklearn.cluster import KMeans
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
df_clean = None
categorical_values = {}
numerical_ranges = {}
feature_columns = []
model_metrics = {}
dataset_stats = {}

@app.context_processor
def inject_global_vars():
    """Inject project info and model metrics into all templates"""
    return {
        'project': PROJECT_INFO,
        'stats': dataset_stats if dataset_stats else {},
        'metrics': model_metrics if model_metrics else {},
        'has_data': df is not None,
        'has_model': model is not None
    }

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
            'std_cost_formatted': f"${df[target].std():,.2f}"
        }
        
        dataset_stats = stats
        return stats
    
    except Exception as e:
        print(f"Error calculating stats: {str(e)}")
        return {}

def load_data_and_model():
    """Load the dataset and trained model - EXACTLY as notebook"""
    global model, df, df_clean, categorical_values, numerical_ranges, feature_columns, model_metrics
    
    try:
        print("\n" + "="*80)
        print("ML-BASED CLASS 5 CONSTRUCTION COST ESTIMATOR")
        print("="*80)
        print("\nLoading and exploring the dataset...")
        
        # Load data
        csv_path = 'data/base_data_for_model.csv'
        if not os.path.exists(csv_path):
            print(f"❌ Error: {csv_path} not found.")
            return False
        
        df = pd.read_csv(csv_path, low_memory=False)
        print(f"✅ Dataset loaded successfully: {df.shape[0]} projects with {df.shape[1]} features.")
        
        # Define the allowed columns (EXACTLY from notebook)
        best_columns = [
            'inflation_factor',
            'total_project_cost_normalized_2025',
            'official_budget_range',
            'ciqs_complexity_category',
            'cnt_division', 
            'cnt_item_code',
            'county_name',
            'area_type',
            'acf',
            'project_latitude',
            'project_longitude',
            'project_type',
            'project_category',
            'project_state'
        ]
        
        # Verify available columns
        available_columns = [col for col in best_columns if col in df.columns]
        print(f"\n✅ Using {len(available_columns)} available columns (out of {len(best_columns)} possible):")
        print(f"Available columns: {', '.join(available_columns)}")
        
        # Define target variable
        target = 'total_project_cost_normalized_2025'
        
        # Calculate dataset statistics
        calculate_dataset_stats()
        
        # DATA PREPROCESSING - EXACTLY as notebook
        print("\n" + "="*80)
        print("DATA PREPROCESSING AND FEATURE ENGINEERING")
        print("="*80)
        
        # Create a copy with only available columns
        df_clean = df[available_columns].copy()
        
        # Handle missing values - EXACTLY as notebook
        missing_before = df_clean.isnull().sum().sum()
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                if df_clean[col].dtype in ['int64', 'float64']:
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                else:
                    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        
        missing_after = df_clean.isnull().sum().sum()
        print(f"Handled {missing_before} missing values. Remaining missing: {missing_after}")
        
        # FEATURE ENGINEERING - EXACTLY as notebook
        print("\nFeature Engineering:")
        print("1. Creating regional clusters from states and geographic coordinates")
        
        # Create region from states using clustering - EXACTLY as notebook
        state_coords = {}
        for state in df_clean['project_state'].unique():
            if 'project_latitude' in df_clean.columns and 'project_longitude' in df_clean.columns:
                state_data = df_clean[df_clean['project_state'] == state]
                lat = state_data['project_latitude'].median()
                lng = state_data['project_longitude'].median()
                if not np.isnan(lat) and not np.isnan(lng):
                    state_coords[state] = (lat, lng)
        
        # Only proceed with clustering if we have coordinates
        if state_coords:
            states = list(state_coords.keys())
            coords = np.array([state_coords[state] for state in states])
            
            optimal_k = min(4, len(coords))
            kmeans = KMeans(n_clusters=optimal_k, random_state=42)
            kmeans.fit(coords)
            
            state_to_region = {}
            for i, state in enumerate(states):
                region = kmeans.predict(np.array([state_coords[state]]).reshape(1, -1))[0]
                state_to_region[state] = f"Region_{region}"
            
            df_clean['region'] = df_clean['project_state'].map(state_to_region)
            print(f"✅ Created {optimal_k} geographic regions based on clustering state coordinates.")
        
        # Prepare features (exclude target and lat/long as in notebook)
        features = [col for col in df_clean.columns if col not in [target, "project_latitude", "project_longitude"]]
        feature_columns = features
        
        print(f"\n✅ Features to use: {features}")
        
        # Store categorical values for dropdowns
        for col in features:
            if df_clean[col].dtype == 'object':
                unique_vals = df_clean[col].dropna().unique().tolist()
                categorical_values[col] = sorted([str(v) for v in unique_vals])
        
        # Store numerical ranges
        for col in features:
            if df_clean[col].dtype in ['int64', 'float64']:
                numerical_ranges[col] = {
                    'min': float(df_clean[col].min()),
                    'max': float(df_clean[col].max()),
                    'mean': float(df_clean[col].mean()),
                    'median': float(df_clean[col].median()),
                    'std': float(df_clean[col].std())
                }
        
        print(f"✅ Loaded {len(categorical_values)} categorical features")
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
                        print(f"✅ Model metrics loaded: R²={model_metrics.get('r2_score', 0):.4f}")
            except Exception as e:
                print(f"⚠️ Could not load model: {str(e)}")
                should_retrain = True
        else:
            should_retrain = True
        
        if should_retrain:
            print("\n🔄 Training new model...")
            model, model_metrics = train_new_model(df_clean, features, target)
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

def train_new_model(df_clean, features, target):
    """Train a new Random Forest model - EXACTLY as notebook"""
    try:
        print("\n" + "="*80)
        print("MODEL DEVELOPMENT")
        print("="*80)
        
        # Prepare features and target
        X = df_clean[features]
        y = df_clean[target]
        
        print(f"Training set preparation...")
        print(f"Features: {features}")
        
        # Split data - EXACTLY as notebook
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Identify categorical and numerical features
        categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
        numerical_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
        
        print(f"\nCategorical features for encoding: {len(categorical_cols)}")
        print(f"Numerical features for scaling: {len(numerical_cols)}")
        
        # Create preprocessing pipeline - EXACTLY as notebook
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ]
        )
        
        # Create pipeline with Random Forest - EXACTLY as notebook
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        # Train model
        print("\n🚀 Training Random Forest...")
        pipeline.fit(X_train, y_train)
        print("✅ Training completed")
        
        # Evaluate
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_percentage_error
        train_mape = mean_absolute_percentage_error(y_train, y_train_pred) * 100
        test_mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        print(f"\n{'='*80}")
        print(f"📈 Model Performance:")
        print(f"{'='*80}")
        print(f"  Train MAPE:   {train_mape:.2f}%")
        print(f"  Test MAPE:    {test_mape:.2f}%")
        print(f"  Test R²:      {test_r2:.4f}")
        print(f"  Test RMSE:    ${test_rmse:,.2f}")
        print(f"  Test MAE:     ${test_mae:,.2f}")
        print(f"{'='*80}\n")
        
        from datetime import datetime
        metrics = {
            'train_mape': float(train_mape),
            'test_mape': float(test_mape),
            'r2_score': float(test_r2),
            'mae': float(test_mae),
            'mae_formatted': f"${test_mae:,.2f}",
            'rmse': float(test_rmse),
            'rmse_formatted': f"${test_rmse:,.2f}",
            'mape_formatted': f"{test_mape:.2f}%",
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features': features,
            'n_features': len(features),
            'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return pipeline, metrics
    
    except Exception as e:
        print(f"❌ Error training: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def get_similar_projects(input_data):
    """Find similar projects in the dataset"""
    global df_clean
    
    if df_clean is None:
        return {}
    
    target = 'total_project_cost_normalized_2025'
    filtered_df = df_clean.copy()
    
    # Filter by key categorical fields
    filter_fields = ['project_type', 'official_budget_range', 'ciqs_complexity_category']
    
    for field in filter_fields:
        if field in input_data and field in df_clean.columns:
            filtered_df = filtered_df[filtered_df[field] == input_data[field]]
    
    # Get statistics
    if len(filtered_df) >= 5:
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
    else:
        # Use overall statistics
        return {
            'count': len(df_clean),
            'avg_cost': float(df_clean[target].mean()),
            'avg_cost_formatted': f"${df_clean[target].mean():,.2f}",
            'median_cost': float(df_clean[target].median()),
            'median_cost_formatted': f"${df_clean[target].median():,.2f}",
            'min_cost': float(df_clean[target].min()),
            'min_cost_formatted': f"${df_clean[target].min():,.2f}",
            'max_cost': float(df_clean[target].max()),
            'max_cost_formatted': f"${df_clean[target].max():,.2f}",
            'std_cost': float(df_clean[target].std()),
            'std_cost_formatted': f"${df_clean[target].std():,.2f}",
            'match_type': 'overall'
        }

# ============= ROUTES =============

@app.route('/')
def home():
    """Home page"""
    return render_template('home.html')

@app.route('/eda')
def eda():
    """EDA page"""
    return render_template('eda.html')

@app.route('/model_comparison')
def model_comparison():
    """Model comparison page"""
    return render_template('model_comparison.html')

@app.route('/dashboard')
def dashboard():
    """Performance dashboard page"""
    return render_template('dashboard.html')

@app.route('/cost_estimator')
def cost_estimator():
    """Cost estimation form page"""
    if df is None or model is None:
        return render_template('error.html', 
                             error='System not ready',
                             message='Dataset or model not loaded.')
    
    return render_template('cost_estimator.html',
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
        
        # Get form data
        input_data = {}
        
        print("\n📋 Form Data Received:")
        for col in feature_columns:
            value = request.form.get(col)
            if value:
                if col in numerical_ranges:
                    try:
                        input_data[col] = float(value)
                        print(f"  ✓ {col}: {value} (numerical)")
                    except ValueError:
                        input_data[col] = numerical_ranges[col]['median']
                        print(f"  ⚠ {col}: using median")
                else:
                    input_data[col] = value
                    print(f"  ✓ {col}: {value} (categorical)")
        
        if not input_data:
            print("❌ ERROR: No input data")
            return jsonify({'success': False, 'error': 'No input data provided'}), 400
        
        # Create DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Fill missing columns
        for col in feature_columns:
            if col not in input_df.columns:
                if col in numerical_ranges:
                    input_df[col] = numerical_ranges[col]['median']
                elif col in categorical_values and categorical_values[col]:
                    input_df[col] = categorical_values[col][0]
        
        # Reorder columns
        input_df = input_df[feature_columns]
        
        print("\n🔮 Making prediction...")
        prediction = model.predict(input_df)[0]
        print(f"✅ PREDICTION: ${prediction:,.2f}")
        
        # Get similar projects
        similar_stats = get_similar_projects(input_data)
        
        from datetime import datetime
        result = {
            'success': True,
            'estimated_cost': float(prediction),
            'estimated_cost_formatted': f"${prediction:,.2f}",
            'confidence_interval': {
                'lower': float(prediction * 0.75),
                'upper': float(prediction * 1.25),
                'lower_formatted': f"${prediction * 0.75:,.2f}",
                'upper_formatted': f"${prediction * 1.25:,.2f}"
            },
            'input_data': input_data,
            'similar_projects': similar_stats,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print("✅ Returning result")
        print("="*60 + "\n")
        
        return jsonify(result)
    
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print("="*60 + "\n")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/data_overview')
def data_overview():
    """Data overview page"""
    return render_template('data_overview.html')

@app.route('/documentation')
def documentation():
    """Documentation page"""
    return render_template('documentation.html')

@app.route('/api/dataset_stats')
def api_dataset_stats():
    """API endpoint for dataset statistics"""
    if not dataset_stats:
        return jsonify({'error': 'Dataset not loaded'}), 404
    return jsonify(dataset_stats)

@app.route('/api/model_metrics')
def api_model_metrics():
    """API endpoint for model metrics"""
    if not model_metrics:
        return jsonify({'error': 'Model metrics not available'}), 404
    return jsonify(model_metrics)

@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', 
                         error='Page not found',
                         message='The page you are looking for does not exist.'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html',
                         error='Internal server error',
                         message='Something went wrong. Please try again later.'), 500

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🏗️  Starting Construction Cost Estimator")
    print("="*80 + "\n")
    
    if load_data_and_model():
        print("\n" + "="*80)
        print("✅ System Ready!")
        print("="*80)
        print("🌐 Access at: http://localhost:5000")
        print("="*80 + "\n")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n" + "="*80)
        print("❌ Initialization failed")
        print("="*80)
        print("Ensure 'data/base_data_for_model.csv' exists")
        print("="*80 + "\n")