import os
from datasets import load_dataset

# Load the dataset
# Login using e.g. `huggingface-cli login` to access the dataset
ds = load_dataset("LaiBenBen/STAT-8307-Group-Project-Gp17-Dataset")

# Create the 'data' folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Save each split of the dataset (train, validation, test) to CSV
for split in ds.keys():
    file_path = os.path.join("data", f"data_merged_original_yuke.csv")
    ds[split].to_csv(file_path, index=False)
    print(f"Saved {split} split to {file_path}")