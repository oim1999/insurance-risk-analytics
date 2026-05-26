"""
data_loader.py
Reusable data loading and preprocessing utilities for the insurance data.
"""
import pandas as pd
import numpy as np


def load_data(filepath):
    """
    Load the insurance dataset from a txt file.

    Parameters:
    -----------
    filepath : str
        Path to the txt file 'data/raw/MachineLearningRating_v3.txt'

    Returns:
    --------
    pd.DataFrame
        Raw insurance data`
    """
    try:
        df = pd.read_csv(filepath, sep='|') 
        print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        raise
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def clean_column_names(df):
    """
    Standardize column names: strip spaces, lowercase, replace spaces with underscores.

    Why this matters: Inconsistent column names cause bugs when you reference them.
    """
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df


def parse_dates(df, date_column='transactionmonth'):
    """
    Convert date columns from string to datetime format.

    Why this matters: You can't do time-series analysis on strings.
    """
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        print(f"Parsed {date_column} to datetime. Missing dates: {df[date_column].isna().sum()}")
    return df


def create_derived_metrics(df):
    """
    Create the two core business metrics: Loss Ratio and Margin.

    Loss Ratio > 1.0 means the company paid more in claims than it collected in premiums (bad).
    Loss Ratio < 0.5 means profitable business (good).
    """
    # Avoid division by zero
    df['loss_ratio'] = np.where(
        df['totalpremium'] == 0,
        np.nan,
        df['totalclaims'] / df['totalpremium']
    )

    df['margin'] = df['totalpremium'] - df['totalclaims']

    print("Created derived metrics: loss_ratio, margin")
    return df


def get_data_summary(df):
    """
    Generate a quick summary of the dataset.
    Returns a dictionary with key statistics.
    """
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'missing_values': df.isnull().sum().sum(),
        'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
        'duplicate_rows': df.duplicated().sum(),
        'date_range': None
    }

    if 'transactionmonth' in df.columns:
        summary['date_range'] = (df['transactionmonth'].min(), df['transactionmonth'].max())

    return summary

def clean_and_handle_missing(df):
    """
    Production missing value handling for ACIS dataset.
    Tailored to the actual missing patterns observed.
    """
    df_clean = df.copy()
    initial_rows = len(df_clean)
    handling_log = []
    
    # ============================================
    # STEP 1: Fix hidden missing values (empty strings)
    # ============================================
    string_cols = df_clean.select_dtypes(include=['object']).columns
    
    for col in string_cols:
        # Replace pure whitespace/empty strings with NaN
        df_clean[col] = df_clean[col].replace(r'^\s*$', np.nan, regex=True)
        # Standardize "Not specified" variations
        if col in ['gender', 'maritalstatus', 'title']:
            df_clean[col] = df_clean[col].replace(
                ['Not specified', 'Not Specified', 'not specified'], 
                np.nan
            )
    
    # ============================================
    # STEP 2: Convert numerical -> categorical (your request)
    # ============================================
    df_clean['cylinders'] = df_clean['cylinders'].astype('category')
    df_clean['numberofdoors'] = df_clean['numberofdoors'].astype('category')
    handling_log.append("Converted 'cylinders' and 'numberofdoors' to categorical")
    
    # ============================================
    # STEP 3: DROP completely empty / near-empty columns
    # ============================================
    # FIX: Use errors='ignore' so it won't crash if columns already dropped
    drop_cols = ['numberofvehiclesinfleet', 'crossborder']
    existing_drop_cols = [col for col in drop_cols if col in df_clean.columns]
    
    if existing_drop_cols:
        df_clean = df_clean.drop(columns=existing_drop_cols)
        handling_log.append(f"Dropped {existing_drop_cols} (empty/near-empty columns)")
    else:
        handling_log.append("Drop columns already removed or not found — skipped")
    
    # ============================================
    # STEP 4: DROP rows with missing vehicle specs (0.055% = 552 rows)
    # ============================================
    vehicle_id_cols = ['make', 'model', 'vehicletype', 'bodytype']
    vehicle_id_cols = [col for col in vehicle_id_cols if col in df_clean.columns]  # Safety check
    
    if vehicle_id_cols:
        vehicle_missing = df_clean[vehicle_id_cols].isnull().any(axis=1)
        rows_before = len(df_clean)
        df_clean = df_clean[~vehicle_missing].copy()
        dropped_vehicle = rows_before - len(df_clean)
        if dropped_vehicle > 0:
            handling_log.append(f"Dropped {dropped_vehicle} rows with missing vehicle specs")
    
    # Drop rows with missing capitaloutstanding (2 rows)
    if 'capitaloutstanding' in df_clean.columns:
        df_clean = df_clean.dropna(subset=['capitaloutstanding'])
    
    # ============================================
    # STEP 5: NEVER impute target variables
    # ============================================
    df_clean = df_clean.dropna(subset=['totalpremium', 'totalclaims'])
    handling_log.append("Ensured no missing in TotalPremium/TotalClaims")
    
    # ============================================
    # STEP 6: CATEGORICAL imputation
    # ============================================
    
    # Binary flags: missing = "No"
    binary_cols = ['rebuilt', 'writtenoff', 'converted']
    for col in binary_cols:
        if col in df_clean.columns:
            missing_count = df_clean[col].isnull().sum()
            if missing_count > 0:
                df_clean[col] = df_clean[col].fillna('No')
                handling_log.append(f"{col}: filled {missing_count} missing with 'No'")
    
    # Gender: sensitive field — never impute with statistics
    if 'gender' in df_clean.columns:
        missing_gender = df_clean['gender'].isnull().sum()
        if missing_gender > 0:
            df_clean['gender'] = df_clean['gender'].fillna('Not Specified')
            handling_log.append(f"gender: filled {missing_gender} missing with 'Not Specified'")
    
    # Other categoricals: fill with "Unknown"
    other_cats = ['maritalstatus', 'newvehicle', 'bank', 'accounttype', 
                  'citizenship', 'title', 'language']
    for col in other_cats:
        if col in df_clean.columns:
            missing_count = df_clean[col].isnull().sum()
            if missing_count > 0:
                df_clean[col] = df_clean[col].fillna('Unknown')
                handling_log.append(f"{col}: filled {missing_count} missing with 'Unknown'")
    
    # ============================================
    # STEP 7: NUMERICAL imputation
    # ============================================
    
    # CustomValueEstimate: 78% missing — FLAG then group-impute
    if 'customvalueestimate' in df_clean.columns:
        df_clean['customvalueestimate_missing'] = df_clean['customvalueestimate'].isnull().astype(int)
        missing_cve = df_clean['customvalueestimate'].isnull().sum()
        
        if missing_cve > 0:
            # Group-impute by Make + VehicleType median
            df_clean['customvalueestimate'] = df_clean.groupby(['make', 'vehicletype'])['customvalueestimate'].transform(
                lambda x: x.fillna(x.median())
            )
            # If still missing (rare make/type combo), use overall median
            overall_median = df_clean['customvalueestimate'].median()
            df_clean['customvalueestimate'] = df_clean['customvalueestimate'].fillna(overall_median)
            
            handling_log.append(f"customvalueestimate: flagged + imputed by Make/VehicleType median ({missing_cve} values)")
    
    # ============================================
    # STEP 8: Recalculate derived metrics properly
    # ============================================
    # Drop existing loss_ratio/margin if they came from CSV
    for col in ['loss_ratio', 'margin']:
        if col in df_clean.columns:
            df_clean = df_clean.drop(columns=[col])
    
    # Recalculate cleanly
    df_clean['loss_ratio'] = np.where(
        df_clean['totalpremium'] == 0,
        np.nan,  # Undefined — don't fake a number
        df_clean['totalclaims'] / df_clean['totalpremium']
    )
    df_clean['margin'] = df_clean['totalpremium'] - df_clean['totalclaims']
    
    handling_log.append("Recalculated loss_ratio and margin (NaN where TotalPremium=0)")
    
    # ============================================
    # AUDIT TRAIL
    # ============================================
    print("=" * 60)
    print("MISSING VALUE HANDLING LOG")
    print("=" * 60)
    for entry in handling_log:
        print(f"  ✓ {entry}")
    
    print(f"\n{'=' * 60}")
    print(f"Initial rows:    {initial_rows:,}")
    print(f"Final rows:        {len(df_clean):,}")
    print(f"Rows removed:      {initial_rows - len(df_clean):,} ({(initial_rows - len(df_clean))/initial_rows*100:.2f}%)")
    print(f"Final columns:     {len(df_clean.columns)}")
    print(f"Remaining missing: {df_clean.isnull().sum().sum()}")
    print("=" * 60)
    
    return df_clean

def prepare_data(filepath):
    """
    Full pipeline: load, clean, parse dates, create metrics, summarize.
    This is the main function you'll call in your notebook.
    and save a copy of data frame as a csv file for easy access in the notebook.
    """
    df = load_data(filepath)
    df = clean_column_names(df)
    df = parse_dates(df)
    df = create_derived_metrics(df)
    df = clean_and_handle_missing(df)
    summary = get_data_summary(df)


    df.to_csv('../data/processed/insurance_data.csv', index=False)

    print("\n=== DATA SUMMARY ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    return df
