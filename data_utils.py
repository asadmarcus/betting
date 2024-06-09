import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

def preprocess_data(df):
    required_columns = [
        'MaxCH', 'PC>2.5', 'MaxAHH', 'MaxCD', 'PC<2.5', 'BWCA', 'BWCD', 
        'B365C>2.5', 'RecentFormHome', 'B365AHH', 'B365D', 'PCAHH', 'AHCh', 
        'MaxA', 'Avg<2.5', 'HR', 'B365CD', 'LBA', 'BbMxAHA', 'B365CH', 
        'B365C<2.5', 'PSD', 'MaxH', 'HTAG', 'B365AHA', 'RecentFormAway', 
        'MaxC<2.5', 'IWCD', 'Date', 'BbAv>2.5', 'HC', 'BbMx<2.5', 'BbAH', 
        'BbAv<2.5', 'IWCH', 'AvgH', 'LBH', 'BbAvA', 'BbMx>2.5', 'HS', 
        'B365<2.5', 'AR', 'Max<2.5', 'AY', 'AHh', 'P>2.5', 'PAHH', 'BbAvAHH', 
        'AvgCD', 'MaxC>2.5', 'VCA', 'AvgCA', 'Bb1X2', 'Referee', 'BbAvD', 
        'IWD', 'HST', 'FTAG', 'AS', 'AST', 'BbMxAHH', 'BbAvAHA', 'B365>2.5', 
        'BWCH', 'FTR_num', 'VCCH', 'PCAHA', 'PSCA', 'IWA', 'BbAHh', 'AvgCH', 
        'AvgD', 'BbAvH', 'BbMxA', 'WHCA', 'BWA', 'WHH', 'B365A', 'P<2.5', 
        'PSH', 'AvgC>2.5', 'VCCA', 'AvgA', 'IWH', 'MaxD', 'AvgCAHH', 'PAHA', 
        'AvgAHH', 'WHCD', 'WHD', 'MaxAHA', 'HY', 'HF', 'VCD', 'WHCH', 
        'Max>2.5', 'BWD', 'LBD', 'MaxCAHH', 'HTR', 'AC', 'PSCD', 'AF', 
        'MaxCAHA', 'B365CA', 'VCH', 'PSCH', 'AvgAHA', 'IWCA', 'B365H', 
        'B365CAHA', 'AvgCAHA', 'AvgC<2.5', 'HTHG', 'BWH', 'BbMxH', 'BbMxD', 
        'Time', 'B365CAHH', 'WHA', 'MaxCA', 'VCCD', 'FTHG', 'BbOU', 'PSA', 
        'Avg>2.5'
    ]

    # Add missing columns with default values
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0

    # Handle preprocessing
    numeric_features = [col for col in required_columns if df[col].dtype in ['int64', 'float64']]
    categorical_features = [col for col in required_columns if df[col].dtype == 'object']

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    X_preprocessed = preprocessor.fit_transform(df)
    return X_preprocessed, preprocessor

def split_data(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y
