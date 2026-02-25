import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from model import SLModel
from sklearn.metrics import accuracy_score
from dataset import get_data_loaders
from save_params import save_params, load_params

CSV_PATH = Path("G:\\Projects\\Python\\Sign Language\\SL-Data\\dataset.csv")
KEYPOINTS_PATH = Path("G:\\Projects\\Python\\Sign Language\\SL-Data\\keypoints")
ENCODER_PATH = Path("G:\\Projects\\Python\\Sign Language\\SL-Data\\label_encoder.pkl")
MODEL_FILE_PATH = Path('models\\SignLang_model.pth')
INPUT_SIZE = 225
BATCH_SIZE = 32
HIDDEN_UNITS = 512
NUM_LAYERS = 2
NUM_NEURONS = 64
DROPOUT = 0.3
LEARNING_RATE = 0.001
EPOCHS = 200

# Setup training device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the data
train_loader, test_loader, label_encoder = get_data_loaders(csv_path= str(CSV_PATH),label_encoder_path= str(ENCODER_PATH),batch_size= BATCH_SIZE, test_size= 0.2)

# Quick inspection of data
print(f"Data loaded successfully!")
print(f"Total unique classes: {len(label_encoder.classes_)}")

# Initialize the model in GPU
NUM_CLASSES = len(label_encoder.classes_) # 2005 in our case
model = SLModel(NUM_CLASSES, INPUT_SIZE, HIDDEN_UNITS, NUM_LAYERS, NUM_NEURONS, DROPOUT).to(device)

# Setup Loss function, Optimizer and Scheduler
loss_fn = nn.CrossEntropyLoss(label_smoothing= 0.1)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
scheduler = ReduceLROnPlateau(optimizer, mode= 'min', patience= 10, min_lr= 1e-7)

# Configuring evaluation function
def evaluate(model: SLModel, test_loader: DataLoader, loss_fn: nn.CrossEntropyLoss, device: str) -> tuple:
    """
    Evaluates the model on a given dataset loader.
    Returns: Average Loss and Accuracy.
    """
    model.eval() # Set model to evaluation mode
    val_loss = 0
    all_preds = []
    all_labels = []

    # Disable gradient calculations for speed and memory efficiency
    with torch.inference_mode():
        for feature, label in test_loader:
            feature, label = feature.to(device), label.to(device)
            
            # Forward pass
            outputs = model(feature)
            loss = loss_fn(outputs, label)
            val_loss += loss.item()
            
            # Calculate accuracy components
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    avg_loss = val_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return (avg_loss, 100*accuracy)

# ---------------------------------------------------------
#                    Training Loop
# ---------------------------------------------------------

def main() -> None:
    worst_loss = float('inf')   #Setting worst case scenario
    
    if MODEL_FILE_PATH.exists():
        print(f"Loading existing weights from {MODEL_FILE_PATH}...")
        model.load_state_dict(load_params(MODEL_FILE_PATH))
        
    for epoch in range(EPOCHS):
        # ---- TRAINING PHASE ----
        model.train()
        running_loss: float = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)   # Move all features and labels to training device
            
            # Forward Pass
            y_pred = model(features)    # Get the predictions
            CCE_loss = loss_fn(y_pred, labels)  # Calculate the loss
            running_loss += CCE_loss.item() #Add up the losses

            # Backpropagation
            optimizer.zero_grad()   #Nullify the grads
            CCE_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  #Normalize to prevent exploding gradients
            optimizer.step()    # Tweak parameters
        
        avg_train_loss = running_loss / len(train_loader)   # Get avg loss per epoch
            
        # ---- VALIDATION PHASE & SCHEDULER ----
        avg_test_loss, accuracy = evaluate(model, test_loader, loss_fn, device)
        scheduler.step(avg_test_loss)
        
        # ---- LOGGING ----
        print(f"Epoch{epoch}/{EPOCHS}: Train Loss: {avg_train_loss:.5f}  ||  Test Loss: {avg_test_loss:.5f}  ||  Accuracy: %{accuracy:.4f}")
        
        # ---- SAVING BEST MODEL PARAMETERS PER TRAINING SESSION ----
        if avg_test_loss < worst_loss:
            worst_loss = avg_test_loss
            save_params(model)
            print(f"Saved parameters to {MODEL_FILE_PATH}")
        print("-" * 30)
        
if __name__ == "__main__":
    main()
            
