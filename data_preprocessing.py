import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

REQUIRED_COLUMNS = {
    'LBA', 'LBD', 'LBH', 'RecentFormHome', 'RecentFormAway', 'Referee', 'FTR_num',
    'PC>2.5', 'AvgAHA', 'AHCh', 'PAHH', 'B365CAHA', 'MaxAHA', 'PAHA', 'AvgCAHH',
    'B365CD', 'AvgCD', 'WHCA', 'MaxCAHH', 'MaxCAHA', 'WHCH', 'B365CH', 'PCAHH',
    'MaxD', 'AvgCA', 'VCCH', 'MaxCA', 'PCAHA', 'Avg>2.5', 'AvgA', 'AvgC>2.5',
    'B365C>2.5', 'PC<2.5', 'WHCD', 'VCCD', 'IWCA', 'B365C<2.5', 'VCCA', 'AvgAHH',
    'AvgD', 'MaxH', 'AvgC<2.5', 'AvgCH', 'Max<2.5', 'Max>2.5', 'MaxCD', 'BWCD',
    'IWCH', 'B365AHA', 'B365AHH', 'Time', 'P>2.5', 'Avg<2.5', 'AvgH', 'P<2.5',
    'B365CA', 'B365>2.5', 'BWCA', 'B365CAHH', 'IWCD', 'MaxC>2.5', 'AvgCAHA',
    'MaxCH', 'MaxA', 'BWCH', 'B365<2.5', 'MaxAHH', 'AHh', 'MaxC<2.5',
    'AF', 'AC', 'HST', 'AST', 'HF', 'HS', 'HC', 'AS',
    'BbAH', 'BbMxAHA', 'BbOU', 'BbAv<2.5', 'BbMx<2.5', 'BbAvAHA', 'Bb1X2', 'BbMxH',
    'BbAv>2.5', 'BbMxAHH', 'BbAvH', 'BbMxD', 'BbAvA', 'BbMx>2.5', 'BbMxA', 'BbAvD',
    'BbAvAHH', 'BbAHh'
}

def load_data(file_path):
    return pd.read_excel(file_path)

def ensure_required_columns(df, required_columns):
    df = df.copy()
    default_values = {col: 0 for col in required_columns}
    default_values.update({'Referee': 'Unknown', 'Time': 0})

    for column in required_columns:
        if column not in df.columns:
            if column == 'FTR_num':
                if 'FTR' in df.columns:
                    df.loc[:, column] = df['FTR'].map({'H': 1, 'D': 0, 'A': -1})
                else:
                    df.loc[:, column] = default_values[column]
            else:
                df.loc[:, column] = default_values[column]
    return df

def preprocess_data(df):
    df = df.dropna().copy()
    df = ensure_required_columns(df, REQUIRED_COLUMNS)

    categorical_features = ['Referee', 'HomeTeam', 'AwayTeam']
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if 'FTR_num' in numerical_features:
        numerical_features.remove('FTR_num')

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    X = df.drop('FTR_num', axis=1)
    y = df['FTR_num']
    X_preprocessed = preprocessor.fit_transform(X)

    return X_preprocessed, y, preprocessor

if __name__ == "__main__":
    data_dir = "./data/2023-2024/"
    df = load_data(data_dir + "all-euro-data-2023-2024.xlsx")

    X_preprocessed, y, preprocessor = preprocess_data(df)
    joblib.dump(preprocessor, './models/preprocessor.pkl')

    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)
    joblib.dump((X_train, X_test, y_train, y_test), './data/split_data.pkl')
