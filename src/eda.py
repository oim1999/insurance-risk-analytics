import pandas as pd

def basic_eda(file_path, sep='|'):
    """
    Loads a delimited text file, displays essential structural metrics,
    performs initial whitespace cleaning, and returns the DataFrame.
    """
    print("="*40)
    print(f"LOADING FILE: {file_path}")
    print("="*40)
    
    
    df = pd.read_csv(file_path, sep=sep)
    
    # Shape and structure
    print(f"\n[INFO] Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} columns\n")
    print("[INFO] Data Types and Non-Null Counts:")
    df.info()
    
    # Missing Value Analysis
    missing_counts = df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]
    
    print("\n[INFO] Missing Values Check:")
    if missing_cols.empty:
        print(" -> No official NaN/Null values found.")
    else:
        print(missing_cols)
        
    # Clean Hidden Whitespaces (Fixes columns like 'Citizenship')
    print("\n[CLEANING] Stripping leading/trailing spaces from text columns...")
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
    
    # Duplicate Check
    duplicate_count = df.duplicated().sum()
    print(f"[INFO] Duplicate Rows Detected: {duplicate_count}")
    
    print("\n" + "="*40)
    print("EDA COMPLETE. DataFrame is ready.")
    print("="*40)
    
    return df
