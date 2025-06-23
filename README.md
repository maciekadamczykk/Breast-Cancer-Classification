# Breast Cancer Classifier

This project is a simple machine learning pipeline for classifying breast cancer data using logistic regression.

## Features
- Loads and preprocesses data from `breast_cancer.csv`
- Splits data into training and test sets
- Scales features using `StandardScaler`
- Trains a logistic regression model
- Evaluates the model with accuracy and confusion matrix
- Visualizes the confusion matrix using Seaborn

## Requirements
- Python 3.x
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

## Usage
1. Place your `breast_cancer.csv` file in the project directory.
2. Install the required packages:
   ```powershell
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```
3. Run the script:
   ```powershell
   python main.py
   ```

## Output
- Prints the number of unique classes in the dataset
- Displays a confusion matrix heatmap
- Prints the accuracy score
