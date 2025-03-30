"""
Contains functions for training and testing a PyTorch model.
"""
import torch
import pandas as pd
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch."""
    
    model.train()  # Set model to training mode
    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device).long()  # Ensure labels are of type LongTensor

        # 1. Forward pass
        y_pred = model(X).logits  # Extract logits from model output

        # 2. Compute loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Zero gradients, Backpropagate, Step optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 4. Compute Accuracy
        y_pred_class = torch.argmax(y_pred, dim=1)
        train_acc += (y_pred_class == y).sum().item()

    # Normalize loss and accuracy
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader.dataset)

    return train_loss, train_acc


def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch."""
    
    model.eval()  # Set model to evaluation mode
    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device).long()  # Ensure labels are of type LongTensor

            # 1. Forward pass
            y_pred = model(X).logits  # Extract logits from model output

            # 2. Compute loss
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            # 3. Compute Accuracy
            y_pred_class = torch.argmax(y_pred, dim=1)
            test_acc += (y_pred_class == y).sum().item()

    # Normalize loss and accuracy
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader.dataset)

    return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          save_path: str = "training_results.csv") -> Dict[str, List]]:
          
    """Trains and tests a PyTorch model."""
    
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    
    model.to(device)

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)

        # Print results per epoch
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

        # Store metrics
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        # Save metrics to CSV
        pd.DataFrame(results).to_csv(save_path, index=False)

    return results

