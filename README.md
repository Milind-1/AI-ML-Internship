# AI-ML-Internship
Data cleaning and preprocessing
This data cleaning and preprocessing project demonstrates essential techniques to prepare raw datasets for machine learning models. Using the popular Titanic survival dataset, we implement industry-standard practices to handle missing data, categorical encoding, feature scaling, and outlier detection – critical steps in any ML pipeline.

Before training predictive models, raw data must be cleaned because,
Missing values can crash ML algorithms, Categorical data needs numerical conversion

Data Exploration & Analysis
1. used to Identify data types (numerical vs categorical)
It checks for missing values per column
It Analyzes statistical distributions

2. Handling Missing Data
Numerical columns: Fills gaps with median values 
Categorical columns: Used mode imputation

3. Categorical Data Encoding
Applies Label Encoding for ordinal categories 
Used One-Hot Encoding for nominal categories 

4. Feature Scaling
Standardized features using Z-score normalization (StandardScaler)
Alternative: Min-Max scaling for [0,1] range (MinMaxScaler)

5. Outlier Management
It detects anomalies with IQR (Interquartile Range) method
It visualizes using boxplots and scatter plots
It removes extreme values impacting model accuracy


The tools that I have used includes:
1. Pandas, NumPy for data Manupulation
2. Matplotlib, Seaborn to Visualize
3. Scikit-learn for ML preprocessing
4. Python 3.9+, Jupyter Notebook


