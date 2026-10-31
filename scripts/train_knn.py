import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from config import TRAIN_CSV, TEST_CSV, MODEL_DIR
from evaluate import get_metrics

# picked from cross-validation: 11 to 31 were tied, 21 is the middle
K = 21

train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)

X_train = train.drop(columns=["Class"])
y_train = train["Class"]
X_test = test.drop(columns=["Class"])
y_test = test["Class"]

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", KNeighborsClassifier(n_neighbors=K)),
])
model.fit(X_train, y_train)

scores = get_metrics(model, X_test, y_test)
for name, value in scores.items():
    print(f"{name}: {value:.4f}")

joblib.dump(model, MODEL_DIR / "knn.joblib")
print("saved model")