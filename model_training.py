import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# Load split data
X_train, X_test, y_train, y_test = joblib.load('./data/split_data.pkl')

# Transform the target labels to start from 0
y_train = y_train + 1
y_test = y_test + 1

models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(),
    "xgboost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    joblib.dump(model, f'./models/{model_name}.pkl')
    
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"{model_name} CV Accuracy: {np.mean(scores):.2f} ± {np.std(scores):.2f}")

for model_name, model in models.items():
    model = joblib.load(f'./models/{model_name}.pkl')
    test_score = model.score(X_test, y_test)
    print(f"{model_name} Test Accuracy: {test_score:.2f}")
