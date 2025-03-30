# train.py

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

def train_models(X_train, y_train):
    logreg = LogisticRegression(penalty='l2', solver='liblinear', random_state=42)
    rf = RandomForestClassifier(n_estimators=40, max_depth=4, class_weight='balanced', random_state=42)
    xgb = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.4, eval_metric='logloss', random_state=42)

    logreg.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    return {'LogisticRegression': logreg, 'RandomForest': rf, 'XGBoost': xgb}
