from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import os
import json
import re
from datetime import datetime, timedelta
# Firebase Admin imports and initialization
try:
    import firebase_admin
    from firebase_admin import credentials, db as firebase_db
    FIREBASE_CREDENTIALS = os.environ.get("FIREBASE_CREDENTIALS")
    FIREBASE_DB_URL = os.environ.get("FIREBASE_DB_URL")
    if FIREBASE_DB_URL and not firebase_admin._apps:
        cred = None
        if FIREBASE_CREDENTIALS:
            # Support either path to JSON file or raw JSON string
            if os.path.isfile(FIREBASE_CREDENTIALS):
                cred = credentials.Certificate(FIREBASE_CREDENTIALS)
            else:
                try:
                    cred = credentials.Certificate(json.loads(FIREBASE_CREDENTIALS))
                except Exception:
                    cred = None
        if cred:
            firebase_admin.initialize_app(cred, {
                'databaseURL': FIREBASE_DB_URL
            })
except Exception as _fb_err:
    # Firebase optional; API should still run without it
    print(f"Firebase init warning: {_fb_err}")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Path to models directory
MODELS_DIR = Path("../tondo_forecasts/models")
FORECASTS_FILE = Path("../tondo_forecasts/tondo_crime_forecasts.json")
HISTORICAL_DATA_FILE = Path("../tondo_crime_data_barangay_41_43_2019_2025.csv")

# Cache for loaded models to improve performance
model_cache = {}

# Robust months parser to accept values like "12", "12 Months", or empty
def parse_months(value):
    if value is None or value == "":
        return 12
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (ValueError, TypeError):
        match = re.search(r"\d+", str(value))
        if match:
            return int(match.group())
        return 12

# Generate forecast robustly across model types (statsmodels, pmdarima)
def generate_forecast(model, months):
    try:
        if hasattr(model, 'forecast'):
            yhat = model.forecast(steps=months)
        elif hasattr(model, 'predict'):
            try:
                yhat = model.predict(n_periods=months)
            except TypeError:
                yhat = model.predict(steps=months)
        elif hasattr(model, 'get_forecast'):
            res = model.get_forecast(steps=months)
            yhat = getattr(res, 'predicted_mean', res)
        else:
            raise AttributeError('Model does not support forecasting')
        arr = np.asarray(yhat, dtype=float)
        # Replace NaNs with previous value or 0
        if np.isnan(arr).any():
            for i in range(arr.size):
                if np.isnan(arr[i]):
                    arr[i] = arr[i-1] if i > 0 and not np.isnan(arr[i-1]) else 0.0
        # Clamp to non-negative counts
        arr = np.clip(arr, 0, None)
        return arr.tolist()
    except Exception as e:
        print(f"Forecast generation error: {e}")
        # Fallback to zeros to avoid breaking UI
        return [0.0] * int(months)

def load_model(crime_type, location):
    """Load a model from disk or cache"""
    key = f"{crime_type}__{location}"
    filename = f"model_{key.replace(' ', '_').replace('/', '_')}.pkl"
    model_path = MODELS_DIR / filename
    
    # Return from cache if available
    if key in model_cache:
        return model_cache[key]
    
    # Load model from disk
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                model_cache[key] = model
                return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    return None

@app.route('/api/forecast', methods=['GET'])
def get_forecast():
    """Generate a forecast for a specific crime type and location"""
    crime_type = request.args.get('crime_type')
    location = request.args.get('location')
    months = parse_months(request.args.get('months', 12))
    
    if not crime_type or not location:
        return jsonify({"error": "Missing crime_type or location parameter"}), 400
    
    model = load_model(crime_type, location)
    if model is None:
        return jsonify({"error": f"Model not found for {crime_type} at {location}"}), 404
    
    try:
        # Generate forecast values robustly
        forecast_values = generate_forecast(model, months)
        
        # Determine forecast starting month based on history
        last_month_dt = None
        if HISTORICAL_DATA_FILE.exists():
            try:
                df = pd.read_csv(HISTORICAL_DATA_FILE)
                df.columns = [c.strip() for c in df.columns]
                df = df[(df['crimeType'] == crime_type) & (df['location'] == location)]
                if not df.empty:
                    df['month'] = pd.to_datetime(df['dateTime']).dt.to_period('M').astype(str)
                    grouped = df.groupby('month').size().reset_index(name='count').sort_values('month')
                    if not grouped.empty:
                        last_month_dt = datetime.strptime(grouped['month'].iloc[-1], "%Y-%m")
            except Exception as e:
                print(f"History processing error: {e}")
        if last_month_dt is None:
            today = datetime.now()
            last_month_dt = datetime(today.year, today.month, 1)
        
        # Create date range for the forecast starting next month after last history
        forecast_dates = []
        for i in range(1, months + 1):
            next_month = last_month_dt + timedelta(days=32 * i)
            next_month = datetime(next_month.year, next_month.month, 1)
            forecast_dates.append(next_month.strftime("%Y-%m-%d"))
        
        return jsonify({
            "crime_type": crime_type,
            "location": location,
            "forecast": {
                "dates": forecast_dates,
                "values": forecast_values
            }
        })
    except Exception as e:
        return jsonify({"error": f"Error generating forecast: {str(e)}"}), 500

@app.route('/api/crime_types', methods=['GET'])
def get_crime_types():
    """Get all available crime types"""
    crime_types = set()
    
    for filename in os.listdir(MODELS_DIR):
        if filename.startswith("model_") and filename.endswith(".pkl"):
            parts = filename[6:-4].split("__")
            if len(parts) >= 1:
                crime_type = parts[0].replace("_", " ")
                crime_types.add(crime_type)
    
    return jsonify({"crime_types": sorted(list(crime_types))})

@app.route('/api/locations', methods=['GET'])
def get_locations():
    """Get allowed locations (restricted to Barangay 41 and Barangay 43)"""
    allowed_locations = {"Barangay 41", "Barangay 43"}
    return jsonify({"locations": sorted(list(allowed_locations))})

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get all available models"""
    models = []
    
    for filename in os.listdir(MODELS_DIR):
        if filename.startswith("model_") and filename.endswith(".pkl"):
            parts = filename[6:-4].split("__")
            if len(parts) >= 2:
                crime_type = parts[0].replace("_", " ")
                location = parts[1].replace("_", " ")
                models.append({
                    "crime_type": crime_type,
                    "location": location,
                    "model_file": filename
                })
    
    return jsonify({"models": models})

@app.route('/api/summary', methods=['GET'])
def get_summary():
    """Get summary information about the forecasts"""
    try:
        with open(FORECASTS_FILE, 'r') as f:
            data = json.load(f)
        
        # Extract summary information
        crime_types = set()
        locations = set()
        
        for key in data.get("forecasts", {}).keys():
            parts = key.split("__")
            if len(parts) >= 2:
                crime_types.add(parts[0])
                locations.add(parts[1])
        
        summary = {
            "total_models": len(data.get("model_files", {})),
            "crime_types": sorted(list(crime_types)),
            "locations": sorted(list(locations)),
            "timestamp": data.get("timestamp")
        }
        
        return jsonify(summary)
    except Exception as e:
        return jsonify({"error": f"Error loading summary: {str(e)}"}), 500

@app.route('/api/all_forecasts', methods=['GET'])
def get_all_forecasts():
    """Get all pre-computed forecasts"""
    try:
        with open(FORECASTS_FILE, 'r') as f:
            data = json.load(f)
        
        return jsonify({"forecasts": data.get("forecasts", {})})
    except Exception as e:
        return jsonify({"error": f"Error loading forecasts: {str(e)}"}), 500

@app.route('/api/report', methods=['POST'])
def create_report():
    """Accept a civilian crime report and push to Firebase Realtime Database.
    Expects JSON body with keys like: crime_type, location, severity, description, reporter_id, latitude, longitude, timestamp(optional).
    """
    try:
        if 'firebase_admin' not in globals() or not FIREBASE_DB_URL or not firebase_admin._apps:
            return jsonify({"error": "Firebase is not configured. Set FIREBASE_CREDENTIALS and FIREBASE_DB_URL environment variables."}), 500
        data = request.get_json(silent=True) or {}
        if not data:
            return jsonify({"error": "Invalid JSON body"}), 400
        # Basic validation
        crime_type = data.get('crime_type') or data.get('crimeType')
        location = data.get('location')
        if not crime_type or not location:
            return jsonify({"error": "Missing required fields: crime_type and location"}), 400
        # Prepare payload
        payload = {
            "crime_type": crime_type,
            "location": location,
            "severity": data.get('severity'),
            "description": data.get('description'),
            "reporter_id": data.get('reporter_id'),
            "latitude": data.get('latitude'),
            "longitude": data.get('longitude'),
            # Use server timestamp if client didn't provide
            "timestamp": data.get('timestamp') or {".sv": "timestamp"},
            "source": data.get('source') or "civilian_app"
        }
        ref = firebase_db.reference('civilian_crime_reports')
        new_ref = ref.push(payload)
        return jsonify({"status": "success", "id": new_ref.key})
    except Exception as e:
        return jsonify({"error": f"Failed to create report: {str(e)}"}), 500

@app.route('/api/reports', methods=['GET'])
def get_reports():
    """Fetch recent civilian crime reports from Firebase Realtime Database.
    Query params: limit (default 50)
    """
    try:
        if 'firebase_admin' not in globals() or not FIREBASE_DB_URL or not firebase_admin._apps:
            return jsonify({"error": "Firebase is not configured. Set FIREBASE_CREDENTIALS and FIREBASE_DB_URL environment variables."}), 500
        limit = int(request.args.get('limit', 50))
        ref = firebase_db.reference('civilian_crime_reports')
        # Order by timestamp if present
        snapshot = ref.order_by_child('timestamp').limit_to_last(limit).get() or {}
        # Convert to list sorted by timestamp
        items = []
        for key, val in snapshot.items():
            val = val or {}
            val['id'] = key
            items.append(val)
        items.sort(key=lambda x: x.get('timestamp', 0))
        return jsonify({"reports": items, "count": len(items)})
    except Exception as e:
        return jsonify({"error": f"Failed to fetch reports: {str(e)}"}), 500

@app.route('/api/visualization', methods=['GET'])
def get_visualization_data():
    """Get data formatted for visualization charts with historical + dashed forecast indicator"""
    crime_type = request.args.get('crime_type')
    location = request.args.get('location')
    months = parse_months(request.args.get('months', 12))
    if not crime_type or not location:
        return jsonify({"error": "Missing crime_type or location parameter"}), 400
    model = load_model(crime_type, location)
    if model is None:
        return jsonify({"error": f"Model not found for {crime_type} at {location}"}), 404
    try:
        # Build historical monthly counts for the selected crime_type and location
        history_labels = []
        history_values = []
        if HISTORICAL_DATA_FILE.exists():
            df = pd.read_csv(HISTORICAL_DATA_FILE)
            df.columns = [c.strip() for c in df.columns]
            df = df[(df['crimeType'] == crime_type) & (df['location'] == location)]
            if not df.empty:
                df['month'] = pd.to_datetime(df['dateTime']).dt.to_period('M').astype(str)
                grouped = df.groupby('month').size().reset_index(name='count')
                grouped = grouped.sort_values('month')
                history_labels = grouped['month'].tolist()
                history_values = grouped['count'].astype(int).tolist()
        # Determine forecast starting month based on history
        if history_labels:
            last_month_dt = datetime.strptime(history_labels[-1], "%Y-%m")
        else:
            today = datetime.now()
            last_month_dt = datetime(today.year, today.month, 1)
        # Generate forecast and future month labels starting next month
        forecast_values = generate_forecast(model, months)
        forecast_labels = []
        start_dt = last_month_dt
        for i in range(1, months + 1):
            next_month = start_dt + timedelta(days=32 * i)
            next_month = datetime(next_month.year, next_month.month, 1)
            forecast_labels.append(next_month.strftime("%Y-%m"))
        # Combine labels
        all_labels = history_labels + forecast_labels
        # Prepare datasets with null padding (None -> null in JSON) and dashed forecast line
        historical_dataset_values = history_values + [None] * months
        forecast_dataset_values = [None] * len(history_labels) + forecast_values
        chart_data = {
            "type": "line",
            "data": {
                "labels": all_labels,
                "datasets": [
                    {
                        "label": "Historical",
                        "data": historical_dataset_values,
                        "borderColor": "rgb(54, 162, 235)",
                        "backgroundColor": "rgba(54, 162, 235, 0.2)",
                        "tension": 0.1,
                        "fill": False
                    },
                    {
                        "label": "Forecast",
                        "data": forecast_dataset_values,
                        "borderColor": "rgb(255, 99, 132)",
                        "backgroundColor": "rgba(255, 99, 132, 0.2)",
                        "tension": 0.1,
                        "fill": False,
                        "borderDash": [8, 6]
                    }
                ]
            },
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "title": {"display": True, "text": "Crime Count"}
                    },
                    "x": {
                        "title": {"display": True, "text": "Month"}
                    }
                },
                "plugins": {
                    "title": {
                        "display": True,
                        "text": f"Crime Forecast (dashed): {crime_type} at {location}"
                    }
                }
            }
        }
        return jsonify({
            "crime_type": crime_type,
            "location": location,
            "chart_config": chart_data,
            "raw_data": {
                "history": {"labels": history_labels, "values": history_values},
                "forecast": {"labels": forecast_labels, "values": forecast_values}
            }
        })
    except Exception as e:
        return jsonify({"error": f"Error generating visualization: {str(e)}"}), 500

@app.route('/')
def index():
    """API documentation"""
    return """
    <html>
        <head>
            <title>Tondo Crime Forecast API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
                h1 { color: #333; }
                h2 { color: #555; margin-top: 30px; }
                code { background-color: #f4f4f4; padding: 2px 5px; border-radius: 3px; }
                pre { background-color: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>Tondo Crime Forecast API</h1>
            <p>This API provides access to ARIMA forecasting models for crime prediction in the Tondo area.</p>
            
            <h2>Endpoints</h2>
            
            <h3>1. Generate Forecast</h3>
            <p><code>GET /api/forecast?crime_type={crime_type}&location={location}&months={months}</code></p>
            <p>Generate a forecast for a specific crime type and location.</p>
            <p>Parameters:</p>
            <ul>
                <li><code>crime_type</code> (required): Type of crime</li>
                <li><code>location</code> (required): Location in Tondo</li>
                <li><code>months</code> (optional): Number of months to forecast (default: 12)</li>
            </ul>
            
            <h3>2. Get Crime Types</h3>
            <p><code>GET /api/crime_types</code></p>
            <p>Get a list of all available crime types.</p>
            
            <h3>3. Get Locations</h3>
            <p><code>GET /api/locations</code></p>
            <p>Get a list of all available locations.</p>
            
            <h3>4. Get Models</h3>
            <p><code>GET /api/models</code></p>
            <p>Get a list of all available models.</p>
            
            <h3>5. Get Summary</h3>
            <p><code>GET /api/summary</code></p>
            <p>Get summary information about the forecasts.</p>
            
            <h3>6. Get All Forecasts</h3>
            <p><code>GET /api/all_forecasts</code></p>
            <p>Get all pre-computed forecasts.</p>
            
            <h3>7. Test Interface</h3>
            <p><a href="/test_api.html">Interactive API Test Page</a></p>
        </body>
    </html>
    """

@app.route('/test_api.html')
def serve_test_page():
    """Serve the test API HTML page"""
    return send_from_directory('.', 'test_api.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)