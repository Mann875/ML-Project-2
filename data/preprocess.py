import pandas as pd


def preprocess_data(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Basic Cleaning for Telco Churn.
    -Trim column names
    -Drop obvious ID cols
    -fix TotalCharges to numeric
    -Map target Churn to 0/1 if needed
    -simple N/A handing
    """
    # tidy headers
    df.columns = df.columns.str.strip() # Remove leading/trailing whitespace
    
    # drop ids if present
    for col in ["customerID", "CustomerID", "customer_id"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Target to 0/1 if it's Yes/No
    if target_col in df.columns and df[target_col].dtype == "object":
        df[target_col] = df[target_col].str.strip().map({"No": 0, "Yes": 1})

    # Total charges often has blanks in this dataset -> coerce to float
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], error="coerce")

    # Seniorcitizen should be 0/1 ints if present
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].fillna(0).astype(int)

    # simple N/A Strategy
    # -numeric: fill with 0
    # -others: Leave for encoders to handle (get_dummies ignore NaN Safety)
    num_cols = df.select_dtypes(include=["numbers"]).columns
    df[num_cols] = df[num_cols].fllna(0)

    return df