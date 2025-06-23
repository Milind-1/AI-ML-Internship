

# Importing  libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

def load_dataset(file_path):

    try:
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully!")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None


def explore_dataset(df):

    print("\n=== Dataset Information ===")
    print(f"Shape: {df.shape} (rows, columns)")
    print("\nData Types:")
    print(df.dtypes)

    print("\n=== Missing Values ===")
    print(df.isnull().sum())

    print("\n=== Descriptive Statistics ===")
    print(df.describe())


def handle_missing_values(df):

    print("\n=== Handling Missing Values ===")

    # Handling numerical col
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"Filled missing values in {col} with median: {median_val}")

    # Handle category col
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"Filled missing values in {col} with mode: {mode_val}")

    return df


def encode_categorical_features(df):

    print("\n=== Encoding Categorical Features ===")

    # Label Encoding for cols
    label_encoder = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() <= 5:
            df[col] = label_encoder.fit_transform(df[col])
            print(f"Label encoded column: {col}")

    # One-Hot Encoding for cols unique val
    df = pd.get_dummies(df, columns=[col for col in df.select_dtypes(include=['object']).columns
                                     if df[col].nunique() > 5])
    print("One-hot encoded remaining categorical columns")

    return df


def scale_numerical_features(df):

    print("\n=== Scaling Numerical Features ===")

    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    print("Standardized numerical features using StandardScaler")

    return df


def detect_and_remove_outliers(df):

    print("\n=== Handling Outliers ===")

    def remove_outliers_iqr(df, col):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    original_size = len(df)

    for col in numerical_cols:
        df = remove_outliers_iqr(df, col)

    removed_count = original_size - len(df)
    print(f"Removed {removed_count} outliers using IQR method")

    return df


def visualize_outliers(df):

    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

    plt.figure(figsize=(15, 10))
    df[numerical_cols].boxplot()
    plt.title("Boxplot of Numerical Features to Detect Outliers")
    plt.xticks(rotation=45)
    plt.show()


def save_cleaned_data(df, output_path):

    try:
        df.to_csv(output_path, index=False)
        print(f"\nCleaned dataset saved to {output_path}")
    except Exception as e:
        print(f"Error saving cleaned dataset: {str(e)}")


def main():
    """Main function to execute the data cleaning pipeline"""
    # Configuration
    input_file = 'titanic.csv'
    output_file = 'cleaned_titanic.csv'


    df = load_dataset(input_file)
    if df is None:
        return


    explore_dataset(df)

    df = handle_missing_values(df)

    df = encode_categorical_features(df)

    visualize_outliers(df)

    df = detect_and_remove_outliers(df)

    df = scale_numerical_features(df)

    print("\n=== Final Cleaned Dataset ===")
    print(f"Final shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())

    save_cleaned_data(df, output_file)


if __name__ == "__main__":
    main()