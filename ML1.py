'''
Milestone 1: Data Preparation for Machine Learning
This script implements comprehensive data cleaning, transformation, and pipeline creation
for bank transaction fraud detection.
'''

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin


bank_df = pd.read_csv("bank_transactions.csv")
target_column = ''  # when we choose the dataset we need to put the name of the target column here
print('\n--- Dataset Overview ---')
print(f'Number of rows: {bank_df.shape[0]}, Number of columns: {bank_df.shape[1]}')
print(f'\nFirst 5 rows:\n{bank_df.head(5)}')


print('\n--- Column Information ---')
print(bank_df.info())

print('\n--- Statistical Summary ---')
print(bank_df.describe())


print('\n--- Missing Values Check ---')
print('Null values in each column:')
print(bank_df.isnull().sum())
print(f'\nTotal missing values: {bank_df.isnull().sum().sum()}')

# Check for duplicates
print('\n--- Duplicate Check ---')
print(f'Number of duplicate rows: {bank_df.duplicated().sum()}')

# Value counts for categorical columns
print('\n--- Categorical Features Distribution ---')
print(f"\nTransactionType distribution:\n{bank_df['TransactionType'].value_counts()}")
print(f"\nAccountID unique values: {bank_df['AccountID'].nunique()}")

# ============================================================================
# SECTION 2: DATA VISUALIZATION AND OUTLIER DETECTION
# ============================================================================
print("\n" + "="*80)
print("SECTION 2: DATA VISUALIZATION AND OUTLIER DETECTION")
print("="*80)

# Identify numerical and categorical columns
num_cols = bank_df.select_dtypes(include=np.number).columns.tolist()
cat_cols = bank_df.select_dtypes(include=['object']).columns.tolist()

print(f'\nNumerical columns ({len(num_cols)}): {num_cols}')
print(f'Categorical columns ({len(cat_cols)}): {cat_cols}')

# Kurtosis analysis [Slide 41  of CapII - Data Fundamentals]
kurtosis=None   
print('\n--- Kurtosis Analysis ---')
for col in num_cols:
    kurtosis = bank_df[col].kurtosis()
    print(f"{col:30s} → Kurtosis: {kurtosis:7.2f}")
    if kurtosis > 3:
        print(f"  → {col} has a leptokurtic distribution, we have more noise and outliers.")

# Boxplots for numerical features (for outlier detection)
print('\n--- Creating boxplots for outlier detection ---')
# Uncomment to visualize:
# bank_df[num_cols].boxplot(figsize=(12, 8))
# plt.title('Boxplot of All Numerical Features')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# Correlation analysis
print('\n--- Correlation Analysis ---')
corr_matrix = bank_df[num_cols].corr() # Practical class 6
print('Correlation matrix:')
print(corr_matrix)

# Uncomment to visualize correlation heatmap:
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
# plt.title("Correlation Matrix of Numerical Features")
# plt.tight_layout()
# plt.show()

# ============================================================================
# SECTION 3: DATA CLEANING
# ============================================================================
print("\n" + "="*80)
print("SECTION 3: DATA CLEANING")
print("="*80)


bank_df_clean = bank_df.copy()
print('\n--- 3.1: Handling Duplicates ---')
duplicates_before = bank_df_clean.duplicated().sum()
bank_df_clean.drop_duplicates(inplace=True) # Pandas method, class 4
print(f'Duplicates removed: {duplicates_before}')
print(f'Rows remaining: {bank_df_clean.shape[0]}')

print('\n--- 3.2: Cleaning Column Names ---')
bank_df_clean.columns = [col.strip().replace(' ', '_') for col in bank_df_clean.columns]
print(f'Cleaned column names: {bank_df_clean.columns.tolist()}')

print('\n--- 3.3: Missing Values Strategy ---')
missing_summary = bank_df_clean.isnull().sum()
print('Missing values per column:')
print(missing_summary[missing_summary > 0])


print('\n--- Dropping columns with >80% missing values ---')
missing_percentage = (bank_df_clean.isnull().sum() / len(bank_df_clean)) * 100
cols_to_drop = missing_percentage[missing_percentage > 80].index.tolist()

if cols_to_drop:
    print(f'Columns to drop (>80% missing): {cols_to_drop}')
    for col in cols_to_drop:
        print(f'  - {col}: {missing_percentage[col]:.2f}% missing')
    bank_df_clean.drop(columns=cols_to_drop, inplace=True)
    print(f'Dropped {len(cols_to_drop)} column(s)')
    print(f'Remaining columns: {bank_df_clean.shape[1]}')
else:
    print('No columns with >80% missing values found')

print('\nStrategy for remaining missing values:')
print('  - Numerical: median imputation (via pipeline)')
print('  - Categorical: most frequent imputation (via pipeline)')


print('\n--- 3.4: Outlier Detection ---')
for col in bank_df_clean.select_dtypes(include=np.number).columns:
    Q1 = bank_df_clean[col].quantile(0.25)
    Q3 = bank_df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = bank_df_clean[(bank_df_clean[col] < lower_bound) | (bank_df_clean[col] > upper_bound)][col]
    print(f'{col:30s}: {len(outliers):5d} outliers detected (kept for now)')
# Standard Scaler will reduce impact of outliers.


# ============================================================================
# SECTION 4: TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "="*80)
print("SECTION 4: TRAIN-TEST SPLIT")
print("="*80)



print(f'\n--- 5.1: Separating Features and Target ---')
print(f'Target column: {target_column}')


X = bank_df_clean.drop(target_column, axis=1)
y = bank_df_clean[target_column]

print(f'Features shape: {X.shape}')
print(f'Target shape: {y.shape}')


class_counts = y.value_counts()
if len(class_counts) == 2:
    majority_class = class_counts.max()
    minority_class = class_counts.min()
    imbalance_ratio = majority_class / minority_class
    print(f'\nClass imbalance ratio: {imbalance_ratio:.2f}:1')
    if imbalance_ratio > 3:
        print(f'  Warning: Significant class imbalance detected!')
        print(f'   Consider using stratified split or resampling techniques.')


print('\n--- 5.2: Creating Train-Test Split (80/20) ---')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

print(f'Training set size: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)')
print(f'Test set size: {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%)')
print(f'\nTraining target distribution:\n{y_train.value_counts()}')
print(f'\nTest target distribution:\n{y_test.value_counts()}')

# ============================================================================
# SECTION 5: FEATURE TYPE ISOLATION
# ============================================================================
print("\n" + "="*80)
print("SECTION 5: FEATURE TYPE ISOLATION")
print("="*80)

# 5.1: Identify numerical and categorical features in X_train
print('\n--- 5.1: Identifying Feature Types ---')
numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

print(f'Numerical features ({len(numerical_features)}): {numerical_features}')
print(f'Categorical features ({len(categorical_features)}): {categorical_features}')

# 6.2: Create separate DataFrames for different feature types
x_train_num = X_train[numerical_features]
x_train_cat = X_train[categorical_features]

print(f'\n--- 6.2: Feature Type DataFrames ---')
print(f'x_train_num shape: {x_train_num.shape}')
print(f'x_train_cat shape: {x_train_cat.shape}')
print(f'\nMissing values in numerical features:\n{x_train_num.isnull().sum()}')
print(f'\nMissing values in categorical features:\n{x_train_cat.isnull().sum()}')

# ============================================================================
# SECTION 6: TRANSFORMATION PIPELINES
# ============================================================================



numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  
    ('scaler', StandardScaler())
])


categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'  # Drop any columns not specified
)



X_train_transformed = preprocessor.fit_transform(X_train)

print(f'Original training shape: {X_train.shape}')
print(f'Transformed training shape: {X_train_transformed.shape}')

# Get feature names after transformation
feature_names = numerical_features.copy()
if len(categorical_features) > 0:
    # Get one-hot encoded feature names
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = list(cat_encoder.get_feature_names_out(categorical_features))
    feature_names.extend(cat_feature_names)

print(f'\nTotal features after transformation: {len(feature_names)}')


X_train_final = pd.DataFrame(X_train_transformed, columns=feature_names)

print('\n--- Training Data Statistics After Transformation ---')
print(X_train_final.describe())


print('\n--- 8.2: Transforming Test Data ---')
X_test_transformed = preprocessor.transform(X_test)

print(f'Original test shape: {X_test.shape}')
print(f'Transformed test shape: {X_test_transformed.shape}')

X_test_final = pd.DataFrame(X_test_transformed, columns=feature_names)




