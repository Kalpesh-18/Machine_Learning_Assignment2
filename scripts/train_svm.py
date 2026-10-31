import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import TRAIN_CSV, TEST_CSV, MODEL_DIR
from evaluate import get_metrics

# cv scores for C = 1, 10 and 100 were tied, so the smallest is used
C = 1

train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)

X_train = train.drop(columns=["Class"])
y_train = train["Class"]
X_test = test.drop(columns=["Class"])
y_test = test["Class"]

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(kernel="rbf", C=C, probability=True, random_state=42)),
])
model.fit(X_train, y_train)

scores = get_metrics(model, X_test, y_test)
for name, value in scores.items():
    print(f"{name}: {value:.4f}")

joblib.dump(model, MODEL_DIR / "svm.joblib")
print("saved model")