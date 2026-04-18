
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import pickle

# Synthetic dataset generation
np.random.seed(42)
data = pd.DataFrame({
    "session_duration": np.random.randint(10, 500, 1000),
    "page_views": np.random.randint(1, 50, 1000),
    "cart_items": np.random.randint(0, 10, 1000),
    "purchase": np.random.randint(0, 2, 1000)
})

data["cart_to_view_ratio"] = data["cart_items"] / (data["page_views"] + 1)

X = data.drop("purchase", axis=1)
y = data["purchase"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = XGBClassifier()
model.fit(X_train, y_train)

preds = model.predict_proba(X_test)[:,1]
print("ROC-AUC:", roc_auc_score(y_test, preds))

with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)
