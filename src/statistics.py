from data_loader import DataLoader

loader = DataLoader()

data = loader.load_csv("data.csv")
labels = loader.load_csv("labels.csv")

print("=" * 60)
print("CANCER AI DATASET STATISTICS")
print("=" * 60)

print(f"Samples : {data.shape[0]}")
print(f"Columns : {data.shape[1]}")

print(f"\nGene Features : {data.shape[1]-1}")

print("\nMissing Values")
print(data.isnull().sum().sum())

print("\nDuplicate Samples")
print(data.duplicated().sum())

print("\nCancer Classes")

print(labels["Class"].value_counts())

print("\nData Types")

print(data.dtypes.value_counts())