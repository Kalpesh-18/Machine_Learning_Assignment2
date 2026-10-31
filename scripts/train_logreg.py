import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from config import TRAIN_CSV, TEST_CSV, MODEL_DIR
from evaluate import get_metrics

train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)

X_train = train.drop(columns=["Class"])
y_train = train["Class"]
X_test = test.drop(columns=["Class"])
y_test = test["Class"]

# scaler and model live in one pipeline, so the saved file includes the scaling
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000)),
])

model.fit(X_train, y_train)

scores = get_metrics(model, X_test, y_test)
for name, value in scores.items():
    print(f"{name}: {value:.4f}")

joblib.dump(model, MODEL_DIR / "logistic_regression.joblib")
print("saved model")