import pandas as pd

# change the filename if yours is different
df = pd.read_excel("/home/s2026aj05468/ML Assignment 2/Machine_Learning_Assignment2/data/DryBeanDataset/Dry_Bean_Dataset.xlsx")

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