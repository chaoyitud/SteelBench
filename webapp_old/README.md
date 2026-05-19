# DEPMAT WP4 TabPFN Web Application

A user-friendly web interface for TabPFN that allows non-programmers to perform machine learning tasks (classification and regression) by simply uploading CSV files.

## Features

- **Easy File Upload**: Drag and drop or click to upload CSV files
- **Automatic Data Processing**: Handles missing values and converts categorical data
- **Two Task Types**:
  - Classification: Predict categories/labels
  - Regression: Predict numerical values
- **Interactive Results**: 
  - Performance metrics
  - Visualizations (confusion matrix for classification, scatter plot for regression)
  - Detailed prediction tables
  - Classification reports
- **No Code Required**: Simple web interface for all users

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   python app.py
   ```

3. **Access the Web App**:
   Open your browser and go to `http://localhost:5000`

## How to Use

1. **Upload CSV File**: Click the upload area or drag and drop your CSV file
2. **Select Task Type**: Choose Classification or Regression based on your target
3. **Choose Target Column**: Select the column you want to predict
4. **Adjust Test Size**: Set the percentage of data to use for testing (default: 30%)
5. **Run Prediction**: Click the button and wait for results

## Data Requirements

- **Format**: CSV file with headers (supports comma, semicolon, tab, or pipe delimiters)
- **Size**: Maximum 100MB file size
- **Dimensions**: Up to 10,000 rows and 500 columns (TabPFN limits)
- **Data Types**: Numerical data preferred (text will be auto-converted)

## Example Datasets

You can test the application with these sample datasets:

### Classification Examples:
- **Iris Dataset**: Predict flower species
- **Breast Cancer Dataset**: Predict cancer diagnosis
- **Wine Quality Dataset**: Predict wine quality categories

### Regression Examples:
- **Boston Housing**: Predict house prices
- **Diabetes Dataset**: Predict diabetes progression
- **Auto MPG**: Predict fuel efficiency

## Understanding Results

### Classification Results:
- **Accuracy**: Percentage of correct predictions
- **ROC AUC**: Area under ROC curve (higher is better)
- **Confusion Matrix**: Visual representation of prediction accuracy
- **Classification Report**: Detailed per-class metrics

### Regression Results:
- **R² Score**: Coefficient of determination (closer to 1 is better)
- **MAE**: Mean Absolute Error (lower is better)
- **MSE**: Mean Squared Error (lower is better)
- **Scatter Plot**: Actual vs Predicted values visualization

## Technical Details

- **Backend**: Flask web framework
- **ML Model**: TabPFN (Tabular Prior-Fitted Network)
- **Frontend**: Bootstrap 5 with JavaScript
- **Visualization**: Matplotlib and Seaborn
- **Data Processing**: Pandas and Scikit-learn

## Limitations

- TabPFN works best with datasets under 10,000 samples and 500 features
- Large files may take longer to process
- Internet connection required for CDN resources (Bootstrap, Font Awesome)
- CPU processing may be slower than GPU for large datasets

## Troubleshooting

**File Upload Issues**:
- Ensure your file is in CSV format
- Check that column headers are present
- Verify file size is under 100MB

**Prediction Errors**:
- Make sure target column contains the values you want to predict
- Check for too many missing values in the dataset
- Ensure sufficient data for train/test split

**Performance Issues**:
- Consider reducing dataset size if processing is slow
- Close other applications to free up memory
- Use smaller test set size for faster processing

## Support

For issues with TabPFN itself, visit the main TabPFN repository.
For web application specific issues, check the console for error messages.