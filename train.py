# train.py

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

def split_data(X, y, test_size=0., random_state=42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def train_models(X_train, y_train, n_estimators=40, max_depth=4):
    logreg = LogisticRegression(penalty='l2', solver='liblinear', random_state=42)
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight='balanced', random_state=42)
    xgb = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.4, eval_metric='logloss', random_state=42)
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    nb = GaussianNB()

    models = {
        'LogisticRegression': logreg.fit(X_train, y_train),
        'RandomForest': rf.fit(X_train, y_train),
        'XGBoost': xgb.fit(X_train, y_train),
        'DecisionTree': dt.fit(X_train, y_train),
        'NaiveBayes': nb.fit(X_train, y_train)
    }

    return models

