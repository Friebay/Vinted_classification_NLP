from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

df = pd.read_csv('final_df_1.csv').drop(columns=['Unnamed: 0'])
df = df.fillna(0.0)
X = df.drop(columns=['cat']).values
y = df['cat'].values

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1234)
train_idx, test_idx = next(sss.split(X, y))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

# https://scikit-learn.org/stable/modules/sgd.html
# https://medium.com/@juanc.olamendy/sgdclassifier-the-powerhouse-for-large-scale-classification-9ae2369d57fb
model = SGDClassifier(
    loss='log_loss',
    random_state=1234,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=5,
    max_iter=50,
    tol=1e-3,
    class_weight='balanced'
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Tikslumas: {accuracy:.4f}")

print("Klasifikavimo rezultatai:")
print(classification_report(y_test, y_pred))

print("Klasifikavimo matrica:")
print(confusion_matrix(y_test, y_pred))