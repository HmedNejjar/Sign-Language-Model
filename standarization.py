import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from tqdm import tqdm

def preprocess_sequence(data, target_frames=60):
    # 1. TEMPORAL STANDARDIZATION (Interpolation)
    current_frames = data.shape[0]
    if current_frames != target_frames:
        x = np.linspace(0, 1, current_frames)
        f = interp1d(x, data, axis=0, kind='linear', fill_value=0)
        new_x = np.linspace(0, 1, target_frames)
        data = f(new_x)

    # 2. SPATIAL NORMALIZATION (Centering)
    reshaped = data.reshape(target_frames, 75, 3) # 225 / 3 = 75 landmarks
    
    for frame_idx in range(target_frames):
        nose = reshaped[frame_idx, 0, :].copy()
        if not np.all(nose == 0):
            reshaped[frame_idx, :, :] -= nose
            
    # 3. FEATURE SCALING (Standardization)
    # Final data should be float32 for PyTorch
    data = reshaped.reshape(target_frames, 225).astype(np.float32)
    
    # Global scaling: ensures values aren't too tiny for the LSTM gradients
    # We avoid dividing by zero if the entire sequence is empty
    std = np.std(data)
    if std > 0:
        data = (data - np.mean(data)) / std
        
    return data

def batch_process_npy(input_dir, output_dir, target_frames=60):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    npy_files = list(input_path.glob("*.npy"))
    print(f"Standardizing and Normalizing {len(npy_files)} files...")
    
    for file_path in tqdm(npy_files):
        try:
            data = np.load(file_path)
            if data.ndim != 2 or data.shape[1] != 225:
                continue
                
            processed_data = preprocess_sequence(data, target_frames)
            np.save(output_path / file_path.name, processed_data)
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

if __name__ == "__main__":
    RAW_DATA_DIR = "G:/Projects/Python/Sign Language/SL-Data/keypoints"
    PROCESSED_DATA_DIR = "G:/Projects/Python/Sign Language/SL-Data/new_keypoints"
    batch_process_npy(RAW_DATA_DIR, PROCESSED_DATA_DIR)