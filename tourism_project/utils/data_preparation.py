import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from huggingface_hub import hf_hub_download
import os

# Define Hugging Face repository details
HUB_REPO_NAME = "tourism-package-prediction"
HF_DATA_FILE = "tourism.csv"
LOCAL_DATA_PATH = "tourism_project/data"

# Ensure local data directory exists
os.makedirs(LOCAL_DATA_PATH, exist_ok=True)

# Download dataset from Hugging Face Hub
try:
    data_file_path = hf_hub_download(
        repo_id=HUB_REPO_NAME,
        filename=HF_DATA_FILE,
        repo_type="dataset",
        local_dir=LOCAL_DATA_PATH,
        local_dir_use_symlinks=False
    )
    print(f"Dataset downloaded to: {data_file_path}")
except Exception as e:
    raise RuntimeError(f"Failed to download dataset from Hugging Face Hub: {e}")

# Load the dataset
df = pd.read_csv(data_file_path)

print("Original DataFrame head:")
print(df.head())
print("\nOriginal DataFrame info:")
df.info()

# Drop CustomerID as it's not needed for modeling
df.drop('CustomerID', axis=1, inplace=True)

# Handle missing values
# Impute numerical columns with median
for col in df.select_dtypes(include=np.number).columns:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)

# Impute categorical columns with mode
for col in df.select_dtypes(include='object').columns:
    if df[col].isnull().any():
        df[col].fillna(df[col].mode()[0], inplace=True)

print("\nDataFrame info after handling missing values:")
df.info()

# Feature Engineering / Encoding
# Convert 'Gender' to numerical (Male: 1, Female: 0)
# Handle potential missing Gender values before mapping if not already handled by mode imputation
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].replace({'Male': 1, 'Female': 0})

# Convert 'Passport' and 'OwnCar' to boolean if not already (0,1)
# They are already 0/1 based on data description, but ensuring consistency
if 'Passport' in df.columns: # Check if column exists before trying to convert
    df['Passport'] = df['Passport'].astype(int)
if 'OwnCar' in df.columns: # Check if column exists before trying to convert
    df['OwnCar'] = df['OwnCar'].astype(int)


# One-hot encode other categorical features
categorical_cols = df.select_dtypes(include='object').columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("\nDataFrame head after encoding:")
print(df.head())

# Define features (X) and target (y)
X = df.drop('ProdTaken', axis=1)
y = df['ProdTaken']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nX_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Ensure model_building directory exists
os.makedirs("tourism_project/model_building", exist_ok=True)

# Save processed data for later use
X_train.to_csv('tourism_project/model_building/X_train.csv', index=False)
X_test.to_csv('tourism_project/model_building/X_test.csv', index=False)
y_train.to_csv('tourism_project/model_building/y_train.csv', index=False)
y_test.to_csv('tourism_project/model_building/y_test.csv', index=False)

print("\nProcessed data saved to tourism_project/model_building/")
