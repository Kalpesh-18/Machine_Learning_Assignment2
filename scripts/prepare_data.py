import pandas as pd
from sklearn.model_selection import train_test_split
from config import RAW_DATA, TRAIN_CSV, TEST_CSV

df = pd.read_excel(RAW_DATA)

# decision: drop exact duplicate rows
before = len(df)
df = df.drop_duplicates()
print("rows dropped as duplicates:", before - len(df))

X = df.drop(columns=["Class"])
y = df["Class"]

# stratify keeps the class proportions the same in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_df = X_train.copy()
train_df["Class"] = y_train
test_df = X_test.copy()
test_df["Class"] = y_test

train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test_data.csv", index=False)

print("train shape:", train_df.shape)
print("test shape:", test_df.shape)
print(test_df["Class"].value_counts())