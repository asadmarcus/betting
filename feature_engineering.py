import pandas as pd

def engineer_features(df):
    # Ensure the 'FTR' column is categorical
    df['FTR'] = df['FTR'].astype('category')

    # Map 'FTR' column to numeric values
    ftr_mapping = {'H': 1, 'D': 0, 'A': -1}
    df['FTR_num'] = df['FTR'].map(ftr_mapping)

    # Ensure numerical columns are properly typed
    numerical_cols = ['FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'FTR_num']
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill missing values in numerical columns
    df[numerical_cols] = df[numerical_cols].fillna(0)

    # Create rolling features for recent form
    df['RecentFormHome'] = df.groupby('HomeTeam')['FTR_num'].rolling(5).sum().reset_index(level=0, drop=True)
    df['RecentFormAway'] = df.groupby('AwayTeam')['FTR_num'].rolling(5).sum().reset_index(level=0, drop=True)

    # Fill any missing values that might result from the rolling function
    df['RecentFormHome'] = df['RecentFormHome'].fillna(0)
    df['RecentFormAway'] = df['RecentFormAway'].fillna(0)

    def calculate_recent_form(df, team_col, form_col, n=5):
    df = df.copy()
    df['RollingForm'] = df.groupby(team_col)[form_col].rolling(window=n).sum().reset_index(level=0, drop=True).fillna(0)
    return df['RollingForm']

    def add_recent_form_columns(df):
    df['RecentFormHome'] = calculate_recent_form(df, 'HomeTeam', 'FTR_num')
    df['RecentFormAway'] = calculate_recent_form(df, 'AwayTeam', 'FTR_num')

    return df

if __name__ == "__main__":
    processed_data_path = "./data/processed/processed_data.csv"
    df = pd.read_csv(processed_data_path)
    df = engineer_features(df)
    df.to_csv(processed_data_path, index=False)
