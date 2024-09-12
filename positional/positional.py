import sys
sys.path.append('./')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  # Import tqdm for progress bars
from modules import Tokenizer, TextDataset, PositionalModel
import os
import matplotlib.pyplot as plt
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to check if targets are valid indices
def check_target_indices(targets, vocab_size):
    if torch.any(targets >= vocab_size) or torch.any(targets < 0):
        raise ValueError(f"Target indices out of range. Vocabulary size: {vocab_size}")

# Training function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    with tqdm(total=len(dataloader), desc="Training", unit="batch") as pbar:
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Check target indices
            check_target_indices(targets, model.vocab_size)
            
            optimizer.zero_grad()
            output = model(inputs)
            
            # Ignore the padding token during loss calculation
            loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item(), batch=batch_idx + 1)
            pbar.update(1)
    
    avg_loss = total_loss / len(dataloader)
    print(f"Training Loss: {avg_loss:.4f}")
    return avg_loss

# Evaluation function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        with tqdm(total=len(dataloader), desc="Evaluating", unit="batch") as pbar:
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                output = model(inputs)
                
                loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))
                
                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item(), batch=batch_idx + 1)
                pbar.update(1)
    
    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss

def main():
    # Load and prepare data
    print("Loading training data...")
    with open('ptbdataset/ptb.train.txt', 'r', encoding='UTF-8') as file:
        train_text = file.readlines()
    print(f"Loaded {len(train_text)} training samples")

    tokenizer = Tokenizer(train_text)
    
    print("Tokenizing training data...")

    print("Loading validation data...")
    with open('ptbdataset/ptb.valid.txt', 'r', encoding='UTF-8') as file:
        val_text = file.readlines()
    print(f"Loaded {len(val_text)} validation samples")

    print("Creating datasets...")
    train_dataset = TextDataset(train_text, tokenizer)
    val_dataset = TextDataset(val_text, tokenizer)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Hyperparameters
    vocab_size = tokenizer.vocab_size()  # Adjust as needed
    d_model = 256
    batch_size = 128
    epochs = 5
    learning_rate = 1e-4

    print("Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    print("Initializing model, loss function, and optimizer...")
    model = PositionalModel(vocab_size, d_model).to(device)
    
    # Use pad token id in tokenizer for ignore_index
    pad_token_id = tokenizer.pad_id()  # You should have this in your tokenizer
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)  # Ignore padding in loss calculation
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    # Training and evaluation loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # Plot the training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    # Save plot as an image
    plt.savefig('positional/training_validation_loss.png')

    # Save the model
    torch.save(model.state_dict(), 'positional/positional_model.pth')
    print("Model saved to decoder_model.pth")


if __name__ == '__main__':
    main()
