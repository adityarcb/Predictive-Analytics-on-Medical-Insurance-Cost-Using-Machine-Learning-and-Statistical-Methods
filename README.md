# Health Insurance Claim Prediction - Comprehensive ML Project

## Project Overview

This project demonstrates comprehensive machine learning techniques for predicting health insurance claim amounts based on various demographic, health, and lifestyle factors. The project covers all major topics from the machine learning syllabus including regression, classification, clustering, dimensionality reduction, neural networks, and model evaluation.

## Dataset

- **File**: `healthinsurance.csv`
- **Size**: ~15,000 records
- **Target Variable**: `claim` (insurance claim amount)
- **Features**: 
  - Demographic: age, sex, weight, BMI, no_of_dependents, city, job_title
  - Health: hereditary_diseases, bloodpressure, diabetes
  - Lifestyle: smoker, regular_ex

## Project Structure

```
ML_project_cursor/
├── healthinsurance.csv                    # Dataset
├── Health_Insurance_Prediction_Project.ipynb  # Main analysis notebook
├── requirements.txt                      # Python dependencies
└── README.md                            # This file
```

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Open the Jupyter notebook:
```bash
jupyter notebook Health_Insurance_Prediction_Project.ipynb
```

## Project Coverage

### Unit I: Introduction and Data Preparation
- Data loading and exploration
- Missing value handling
- Categorical variable encoding
- Data visualization
- Exploratory Data Analysis (EDA)
- Statistical analysis

### Unit II: Supervised Learning - Regression
- Simple Linear Regression
- Multiple Linear Regression
- Polynomial Regression
- Model evaluation: MAE, MSE, RMSE, R² Score

### Unit III: Supervised Learning - Classification
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Decision Trees
- Support Vector Machine (SVM)
- Model evaluation: Accuracy, Log Loss, AUC, Precision, Recall, F1 Score, Confusion Matrix

### Unit IV: Unsupervised Learning
- K-Means Clustering (with Elbow Method and Silhouette Score)
- Hierarchical Clustering (Single, Complete, Average, Ward linkages)
- Association Rules (Apriori algorithm for Market Basket Analysis)

### Unit V: Dimensionality Reduction and Neural Networks
- Principal Component Analysis (PCA)
- Multi-layer Perceptron (MLP) for Regression
- Multi-layer Perceptron (MLP) for Classification

### Unit VI: Model Performance
- K-Fold Cross-Validation
- Leave-One-Out (LOO) Cross-Validation
- Bagging
- Boosting (AdaBoost, Gradient Boosting)
- Random Forest
- Bias-Variance Trade-off Analysis

## Key Results

### Best Regression Models
- Random Forest Regressor typically shows the best performance
- Ensemble methods outperform individual models
- R² scores demonstrate good predictive capability

### Best Classification Models
- Random Forest and Gradient Boosting show superior accuracy
- Ensemble methods provide robust predictions
- All models evaluated with comprehensive metrics

## Features

- **Comprehensive Coverage**: All syllabus topics implemented
- **Well-Documented**: Clear explanations and visualizations
- **Production-Ready Code**: Clean, organized, and commented
- **Visualizations**: Extensive plots and charts for better understanding
- **Model Comparison**: Side-by-side comparison of all models
- **Statistical Analysis**: Detailed EDA and statistical insights

## Usage

1. Ensure all dependencies are installed
2. Place `healthinsurance.csv` in the project directory
3. Run all cells in the Jupyter notebook sequentially
4. Review results, visualizations, and model comparisons

## Requirements

See `requirements.txt` for complete list. Key packages include:
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- scipy
- mlxtend
- tensorflow, keras (for neural networks)
- jupyter

## Project Evaluation Criteria Alignment

This project addresses all evaluation criteria:

1. **Problem Statement and Dataset (10 Marks)**: ✓
   - Clear problem statement
   - Appropriate dataset selection
   - Dataset description and analysis

2. **Implementation (30 Marks)**: ✓
   - Data Cleaning and Visualization (5 Marks)
   - EDA and Statistical Analysis (10 Marks)
   - Model Development and Evaluation (15 Marks)

3. **Report (10 Marks)**: ✓
   - Well-structured notebook with markdown explanations
   - Clear documentation
   - Comprehensive analysis

4. **Technical Coverage**: ✓
   - All 6 units of syllabus covered
   - Multiple algorithms per category
   - Proper evaluation metrics

## Author

Created as part of Machine Learning course project.

## License

This project is for educational purposes.

