"""
DEPMAT WP4 TabPFN Web Application

A simple web interface for TabPFN that allows non-programmers to perform
regression and classification tasks by uploading CSV files.
"""

import os
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

from tabpfn import TabPFNClassifier, TabPFNRegressor
from sklearn.datasets import fetch_openml

def read_csv_with_delimiter_detection(file):
    """
    Automatically detect CSV delimiter and read the file.
    Tries common delimiters: comma, semicolon, tab, pipe.
    """
    # Reset file pointer to beginning
    file.seek(0)
    
    # Read first few lines to detect delimiter
    sample_lines = []
    for i in range(min(5, 100)):  # Read up to 5 lines or first 100 chars
        line = file.readline()
        if isinstance(line, bytes):
            line = line.decode('utf-8', errors='ignore')
        if not line:
            break
        sample_lines.append(line.strip())
    
    # Reset file pointer again
    file.seek(0)
    
    # Common delimiters to try
    delimiters = [',', ';', '\t', '|']
    
    best_delimiter = ','
    max_columns = 0
    
    # Try each delimiter and see which gives the most consistent column count
    for delimiter in delimiters:
        try:
            # Count columns in each sample line
            column_counts = []
            for line in sample_lines:
                if line.strip():  # Skip empty lines
                    columns = line.split(delimiter)
                    column_counts.append(len(columns))
            
            if column_counts:
                # Check if column count is consistent and reasonable (>1 column)
                avg_columns = sum(column_counts) / len(column_counts)
                column_variance = sum((x - avg_columns) ** 2 for x in column_counts) / len(column_counts)
                
                # Prefer delimiter with more columns and low variance
                if avg_columns > max_columns and column_variance < 1.0 and avg_columns > 1:
                    max_columns = avg_columns
                    best_delimiter = delimiter
        except:
            continue
    
    # Reset file pointer one more time before reading with pandas
    file.seek(0)
    
    try:
        # Try reading with the detected delimiter
        df = pd.read_csv(file, sep=best_delimiter, encoding='utf-8')
        
        # Validation: check if we got reasonable data
        if len(df.columns) <= 1 or len(df) == 0:
            raise ValueError("Could not properly parse CSV file")
            
        return df
    except Exception as e:
        # Fallback: try with pandas' built-in delimiter detection
        file.seek(0)
        try:
            df = pd.read_csv(file, sep=None, engine='python', encoding='utf-8')
            return df
        except:
            # Last resort: assume comma delimiter
            file.seek(0)
            df = pd.read_csv(file, sep=',', encoding='utf-8')
            return df

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Create upload directory
UPLOAD_FOLDER = Path('uploads')
UPLOAD_FOLDER.mkdir(exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if using example dataset or uploaded file
        use_example = request.form.get('use_example', 'false').lower() == 'true'
        
        if use_example:
            # Load example dataset
            dataset_name = request.form.get('example_dataset', 'boston_housing')
            if dataset_name == 'boston_housing':
                # Load Boston Housing data using OpenML
                openml_data = fetch_openml(data_id=531, as_frame=True)
                X = openml_data.data
                y = openml_data.target.astype(float)
                df = X.copy()
                df['MEDV'] = y
            else:
                return jsonify({'error': 'Unknown example dataset'})
        else:
            # Check if file was uploaded
            if 'file' not in request.files:
                return jsonify({'error': 'No file uploaded'})
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'})
            
            # Read the CSV file with automatic delimiter detection
            try:
                df = read_csv_with_delimiter_detection(file)
            except Exception as e:
                return jsonify({'error': f'Error reading CSV file: {str(e)}'})
        
        # Get task type and target column
        task_type = request.form.get('task_type')
        target_column = request.form.get('target_column')
        test_size = float(request.form.get('test_size', 0.3))
        
        if not task_type or not target_column:
            return jsonify({'error': 'Please specify task type and target column'})
        
        # Validate target column exists
        if target_column not in df.columns:
            return jsonify({'error': f'Target column "{target_column}" not found in CSV'})
        
        # Prepare data
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Check for non-numeric data and convert if necessary
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    X[col] = pd.to_numeric(X[col])
                except:
                    # For categorical data, use label encoding
                    X[col] = pd.Categorical(X[col]).codes
        
        # Handle missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.mean() if task_type == 'regression' else y.mode()[0] if len(y.mode()) > 0 else 0)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Initialize and train model
        if task_type == 'classification':
            model = TabPFNClassifier(device='cpu')  # Force CPU to avoid Metal issues
            model.fit(X_train, y_train)
            
            # Get predictions and probabilities
            predictions = model.predict(X_test)
            try:
                probabilities = model.predict_proba(X_test)
            except:
                probabilities = None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, predictions)
            
            # Generate classification report
            report = classification_report(y_test, predictions, output_dict=True)
            
            # ROC AUC (if probabilities available)
            roc_auc = None
            if probabilities is not None:
                try:
                    if len(np.unique(y)) == 2:  # Binary classification
                        roc_auc = roc_auc_score(y_test, probabilities[:, 1])
                    else:  # Multiclass
                        roc_auc = roc_auc_score(y_test, probabilities, multi_class='ovr')
                except:
                    pass
            
            # Generate confusion matrix plot
            cm_plot = generate_confusion_matrix_plot(y_test, predictions)
            
            results = {
                'task_type': 'classification',
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'classification_report': report,
                'predictions': predictions.tolist(),
                'actual': y_test.tolist(),
                'confusion_matrix_plot': cm_plot,
                'feature_names': X.columns.tolist(),
                'n_samples': len(X),
                'n_features': len(X.columns),
                'n_train_samples': len(X_train),
                'n_test_samples': len(X_test)
            }
            
        else:  # regression
            model = TabPFNRegressor(device='cpu')  # Force CPU to avoid Metal issues
            model.fit(X_train, y_train)
            
            # Get predictions
            predictions = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            # Generate scatter plot
            scatter_plot = generate_regression_plot(y_test, predictions)
            
            results = {
                'task_type': 'regression',
                'mse': mse,
                'mae': mae,
                'r2_score': r2,
                'predictions': predictions.tolist(),
                'actual': y_test.tolist(),
                'regression_plot': scatter_plot,
                'feature_names': X.columns.tolist(),
                'n_samples': len(X),
                'n_features': len(X.columns),
                'n_train_samples': len(X_train),
                'n_test_samples': len(X_test)
            }
        
        return jsonify({'success': True, 'results': results})
        
    except Exception as e:
        error_msg = str(e)
        traceback_msg = traceback.format_exc()
        return jsonify({'error': f'Error processing request: {error_msg}', 'traceback': traceback_msg})

@app.route('/load_example', methods=['POST'])
def load_example():
    """Load example dataset (Boston Housing)"""
    try:
        dataset_name = request.json.get('dataset', 'boston_housing')
        
        if dataset_name == 'boston_housing':
            # Load Boston Housing data using OpenML
            df = fetch_openml(data_id=531, as_frame=True)  # Boston Housing dataset
            X = df.data
            y = df.target.astype(float)  # Ensure target is float for regression
            
            # Combine features and target
            full_df = X.copy()
            full_df['MEDV'] = y  # MEDV is the target (median home value)
            
            # Sample first 50 rows for preview
            sample_df = full_df.head(50)
            
            return jsonify({
                'success': True,
                'columns': full_df.columns.tolist(),
                'sample_data': sample_df.head(5).to_dict('records'),
                'n_rows': len(full_df),
                'n_columns': len(full_df.columns),
                'dataset_name': 'Boston Housing',
                'description': 'Predict median home values (MEDV) in Boston suburbs based on 13 features',
                'task_type': 'regression',
                'target_column': 'MEDV',
                'csv_data': full_df.to_csv(index=False)
            })
        else:
            return jsonify({'error': 'Unknown dataset'})
            
    except Exception as e:
        return jsonify({'error': f'Error loading example dataset: {str(e)}'})

@app.route('/get_columns', methods=['POST'])
def get_columns():
    """Get column names from uploaded CSV file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        # Read just the header to get column names with delimiter detection
        df_full = read_csv_with_delimiter_detection(file)
        total_rows = len(df_full)  # Get actual row count before truncating
        df_sample = df_full.head(5)  # Keep only first 5 rows for sample data
        
        columns = df_full.columns.tolist()
        sample_data = df_sample.to_dict('records')
        
        return jsonify({
            'success': True, 
            'columns': columns, 
            'sample_data': sample_data,
            'n_rows': total_rows,
            'n_columns': len(columns)
        })
        
    except Exception as e:
        return jsonify({'error': f'Error reading CSV file: {str(e)}'})

def generate_confusion_matrix_plot(y_true, y_pred):
    """Generate base64 encoded confusion matrix plot"""
    try:
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=np.unique(y_true), 
                   yticklabels=np.unique(y_true))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Save to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
        img_buffer.seek(0)
        img_data = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return img_data
    except Exception as e:
        print(f"Error generating confusion matrix plot: {e}")
        return None

def generate_regression_plot(y_true, y_pred):
    """Generate base64 encoded regression scatter plot"""
    try:
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.6)
        
        # Add perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.grid(True, alpha=0.3)
        
        # Save to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
        img_buffer.seek(0)
        img_data = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return img_data
    except Exception as e:
        print(f"Error generating regression plot: {e}")
        return None

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)