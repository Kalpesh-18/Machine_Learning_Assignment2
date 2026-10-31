import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier

from config import TRAIN_CSV, TEST_CSV, MODEL_DIR
from evaluate import get_metrics

MAX_DEPTH = 8

train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)

X_train = train.drop(columns=["Class"])
y_train = train["Class"]
X_test = test.drop(columns=["Class"])
y_test = test["Class"]

model = DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=42)
model.fit(X_train, y_train)

scores = get_metrics(model, X_test, y_test)
for name, value in scores.items():
    print(f"{name}: {value:.4f}")

joblib.dump(model, MODEL_DIR / "decision_tree.joblib")
print("saved model")