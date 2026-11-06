import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

MONTH_ORDER = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]

def load_and_preprocess(csv_path="data/hotel_bookings.csv"):
    df = pd.read_csv(csv_path)
    
    # Clean nulls
    df["country"] = df["country"].fillna("Unknown")
    df["children"] = df["children"].fillna(0).astype(int)
    
    # Derived attributes
    df["total_stays"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["total_guests"] = df["total_guests"].clip(lower=1, upper=10)
    df["adr"] = df["adr"].clip(lower=0, upper=500)
    df["lead_time"] = df["lead_time"].clip(upper=450)
    
    # Ensure correct month categorical ordering
    df["arrival_date_month"] = pd.Categorical(df["arrival_date_month"], categories=MONTH_ORDER, ordered=True)
    
    return df

def train_or_load_model(df, model_path="src/model.pkl"):
    feature_cols = [
        "lead_time", "adr", "total_stays", "total_guests",
        "previous_cancellations", "booking_changes", "total_of_special_requests",
        "is_repeated_guest"
    ]
    
    # Use a high-quality sample for fast responsiveness
    sample = df.sample(min(25000, len(df)), random_state=42)
    X = sample[feature_cols].copy()
    y = sample["is_canceled"]
    
    rf = RandomForestClassifier(n_estimators=60, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    return rf, feature_cols
