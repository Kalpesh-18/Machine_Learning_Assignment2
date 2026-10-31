import pandas as pd
from config import RAW_DATA, TRAIN_CSV, TEST_CSV

df = pd.read_excel(RAW_DATA)

print("shape:", df.shape)
print()
print(df.dtypes)
print()
print("missing values per column:")
print(df.isnull().sum())
print()
print("duplicate rows:", df.duplicated().sum())
print()
print("class counts:")
print(df["Class"].value_counts())
print()
print(df.describe().T)