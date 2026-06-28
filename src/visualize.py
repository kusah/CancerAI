from data_loader import DataLoader
import matplotlib.pyplot as plt

loader = DataLoader()
labels = loader.load_csv("labels.csv")

class_counts = labels["Class"].value_counts()

plt.figure(figsize=(9,6))

bars = plt.bar(class_counts.index, class_counts.values)

plt.title("Distribution of Cancer Types", fontsize=16)

plt.xlabel("Cancer Type", fontsize=12)

plt.ylabel("Number of Samples", fontsize=12)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 3,
        str(int(height)),
        ha='center'
    )

plt.tight_layout()

plt.savefig("reports/cancer_distribution.png", dpi=300)

plt.show()