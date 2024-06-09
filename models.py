from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

def load_model(model_name):
    return joblib.load(f'./models/{model_name}.pkl')

def load_preprocessor():
    return joblib.load('./models/preprocessor.pkl')

def predict(model, preprocessor, data):
    preprocessed_data = preprocessor.transform(data)
    return model.predict_proba(preprocessed_data)


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def get_model(model_name):
    if model_name == 'Logistic Regression':
        return LogisticRegression()
    elif model_name == 'Random Forest':
        return RandomForestClassifier()
    elif model_name == 'XGBoost':
        return XGBClassifier()
    elif model_name == 'KNN':
        return KNeighborsClassifier()
    elif model_name == 'SVM':
        return SVC(probability=True)
    elif model_name == 'Decision Tree':
        return DecisionTreeClassifier()
    elif model_name == 'Naive Bayes':
        return GaussianNB()
    elif model_name == 'Ensemble':
        return VotingClassifier(estimators=[
            ('lr', LogisticRegression()), 
            ('rf', RandomForestClassifier()), 
            ('xgb', XGBClassifier())], voting='soft')

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, precision, recall, f1
