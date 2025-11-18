from hparams import *
from tqdm import tqdm

import numpy as np
from dataloader import *
from model import *

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from hparams_tuning import plot_confusion_matrix


def train_epoch(model: LeNet5, loader: DataLoader, criterion, optimizer, device=DEVICE):
    """Train model for one epoch with gradient clipping

    Args:
        model (LeNet5): Model to train
        loader (DataLoader): Training dataloader
        criterion: Loss function (CrossEntropyLoss)
        optimizer: Optimizer (Adam)
        device (torch.device): Device for training. Defaults to DEVICE.

    Returns:
        tuple: Average loss and accuracy (%) for the epoch
    """
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, desc='Training', leave=False)
    for images, labels in loop:

        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()

        # deal with exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        # tqdm loop
        loop.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def evaluate(model: LeNet5, loader: DataLoader, criterion, device: torch.device):
    """Evaluate model on validation/test set without gradient updates

    Args:
        model (LeNet5): Model to evaluate
        loader (DataLoader): Validation/test dataloader
        criterion: Loss function
        device (torch.device): Device for evaluation

    Returns:
        tuple: Average loss and accuracy (%)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, desc='Evaluating', leave=False)
    with torch.no_grad():
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            # tqdm loop
            loop.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def train_model(model: LeNet5, criterion, optimizer, epochs: int, is_pruning: bool = False, print_info: bool = False):
    """Train model with early stopping and save best checkpoint

    Args:
        model (LeNet5): Model to train
        criterion: Loss function
        optimizer: Optimizer
        epochs (int): Maximum number of training epochs
        is_pruning (bool, optional): Use 'model_pruned_' filename prefix. Defaults to False.
        print_info (bool, optional): Print dataset info. Defaults to False.

    Returns:
        LeNet5: Trained model
    """
    best_acc = 0.0  # stores best accuracy throughout all training
    patience_counter = 0  # counter for early stopping

    train_loader, test_loader = dataset_load(
        epochs=epochs, print_info=print_info)

    loop = tqdm(range(epochs), desc="Epochs", leave=False)
    for epoch in loop:
        _, _ = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        _, test_acc = evaluate(model, test_loader, criterion, DEVICE)

        if test_acc > best_acc:
            best_acc = test_acc

            if is_pruning:
                model_save(
                    model, filename=f"model_pruned_{best_acc:.2f}.pth")
            else:
                model_save(
                    model, filename=f"model_{best_acc:.2f}.pth")

            # tqdm loop - better model saved
            patience_counter = 0
            loop.set_description(
                f'Epochs (Best: {best_acc:.2f}%)')
        else:
            patience_counter += 1

            # tqdm loop - no improvement
            loop.set_description(
                f'Epochs (Best: {best_acc:.2f}%, No improvement: {patience_counter}/{PATIENCE})')
        if patience_counter >= PATIENCE:
            print(
                f"Early stopping - no improvement over {PATIENCE} epochs.")
            break

    if patience_counter >= PATIENCE:
        print(f"Training stopped early at epoch {epoch + 1}/{EPOCHS}")
    else:
        print("Training finished.")

    print(f" Best Test Accuracy: {best_acc:.2f}%")

    return model


def main() -> int:

    model, criterion, optimizer = model_set()
    model = train_model(model, criterion, optimizer, EPOCHS, False)

    return 0


if __name__ == "__main__":
    main()
