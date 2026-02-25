import pandas as pd
import numpy as np
import torch
import joblib
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from typing import cast

class SignLangDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, label_encoder: LabelEncoder) -> None:
        super().__init__()
        self.df = dataframe
        self.label_encoder = label_encoder
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> tuple:
        # Get path and word from dataframe Row
        row = self.df.iloc[idx]
        keypoints_path = row['keypoint']
        word = str(row["word"]).strip()
        
        # Load the array and make it float32 for PyTorch compatibility
        try:
            keypoints = np.load(keypoints_path).astype(np.float32)
        except Exception as e:
            # Fallback in case of corruption
            print(f"Error loading {keypoints_path}: {e}")
            keypoints = np.zeros((60, 225), dtype=np.float32)
            
        # Convert string label to integer    
        try:
            encoded = cast(np.ndarray, self.label_encoder.transform([word]))
            label = int(encoded[0])
        except ValueError as e:
            print(f"Error with word at index {idx}: '{word}' | Row data: {row}")
            raise
        
        # Returns a tuple of (keypoint, corresponing label (converted Text) )
        return (torch.tensor(keypoints), torch.tensor(label, dtype=torch.long))

def get_data_loaders(csv_path:str, label_encoder_path: str, batch_size: int= 32, test_size: float= 0.2) -> tuple:
    # Load the csv file
    df = pd.read_csv(csv_path)
    
    # Set the label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(df['word'].unique())  # We fit on all unique words to ensure every class gets a number
    
    # Save the encoder for decoding the classes
    joblib.dump(label_encoder, label_encoder_path)
    
    # Set the training and testing dataframes
    train_df, test_df = train_test_split(df, test_size= test_size, shuffle= True, random_state= 9)
    
    # Initialize the datasets
    train_ds, test_ds = SignLangDataset(train_df, label_encoder), SignLangDataset(test_df, label_encoder)
    
    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size, shuffle=True)
    
    return (train_loader, test_loader, label_encoder)
    
if __name__ == "__main__":
    # Test the script
    CSV_FILE = "G:\\Projects\\Python\\Sign Language\\SL-Data\\dataset.csv"
    LABEL_PATH = "G:\\Projects\\Python\\Sign Language\\SL-Data\\label_encoder.pkl"
    train_loader, test_loader, encoder = get_data_loaders(CSV_FILE, LABEL_PATH)
    
    print(f"Number of classes: {len(encoder.classes_)}")
    
    # Pull one batch to verify shapes
    features, labels = next(iter(train_loader))
    print(f"Features batch shape: {features.shape}") # Expect [32, 60, 225]
    print(f"Labels batch shape: {labels.shape}")     # Expect [32]
    