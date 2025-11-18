import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import warnings
import os
import time

from hparams import *
from dataloader import *

import math
import torch.nn.utils.prune as prune
from tqdm import tqdm


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5,
                               padding=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x, return_features=False):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        features = F.relu(self.fc2(x))
        x = self.fc3(features)

        if return_features:
            return x, features
        return x


def model_set(lrate: float = LEARNING_RATE, print_info: bool = False) -> tuple[LeNet5, nn.CrossEntropyLoss, optim.Adam]:
    """Initialize LeNet5 model, loss function and optimizer

    Args:
        lrate (float, optional): Learning rate for Adam optimizer. Defaults to LEARNING_RATE.
        print_info (bool, optional): Print model architecture details. Defaults to False.

    Returns:
        tuple[LeNet5, nn.CrossEntropyLoss, optim.Adam]: Model, criterion and optimizer
    """
    model = LeNet5(num_classes=NUM_CLASSES).to(DEVICE)
    if print_info:
        print(f"Model LeNet5 on: {DEVICE}")
        print(
            f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=lrate)
    criterion = nn.CrossEntropyLoss()  # loss function for classification

    if print_info:
        print(f"Loss function: CrossEntropyLoss")
        print(f"Optimizer LR:{lrate}")

    return model, criterion, optimizer


def model_save(model: LeNet5,  filename: str = "model.pth", path: str = MODELS_DIR, device=DEVICE):
    """Save model state_dict to file

    Args:
        model (LeNet5): Model to save
        filename (str, optional): Output filename. Defaults to "model.pth".
        path (str, optional): Directory path for saving. Defaults to MODELS_DIR.
        device (_type_, optional): Device type (unused). Defaults to DEVICE.
    """
    torch.save(model.state_dict(), f"{path}{filename}")


def model_load(path: str = MODEL_PATH, device=DEVICE) -> LeNet5:
    """Load model weights from file and return LeNet5 instance

    Args:
        path (str, optional): Path to .pth model file. Defaults to MODEL_PATH.
        device (_type_, optional): Device to load model on (cpu/cuda). Defaults to DEVICE.

    Raises:
        FileNotFoundError: If model file does not exist

    Returns:
        LeNet5: Loaded model in eval mode
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    model = LeNet5()

    model.load_state_dict(torch.load(
        path, map_location=device, weights_only=False))
    model = model.to(device)
    model.eval()

    return model


def model_load_quantized(path: str, device='cpu', print_info: bool = False):
    """Load dynamically quantized model (int8) from file

    Args:
        path (str): Path to quantized model file
        device (str, optional): Device (forced to 'cpu'). Defaults to 'cpu'.
        print_info (bool, optional): Print quantization info (unused). Defaults to False.

    Raises:
        FileNotFoundError: If model file does not exist

    Returns:
        LeNet5: Quantized model in eval mode on CPU
    """
    if not os.path.exists(path):
        raise FileNotFoundError("Model file not found: {path}")

    # Quantized models MUST be on CPU
    device = 'cpu'

    model_fp32 = LeNet5()
    model_fp32.eval()

    model_quantized = torch.quantization.quantize_dynamic(
        model_fp32,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning, message=".*TypedStorage is deprecated.*")
        model_quantized.load_state_dict(torch.load(
            path, map_location='cpu', weights_only=False))

    # Quantized models stay on CPU (no GPU support for dynamic quantization)
    model_quantized.eval()

    return model_quantized


def model_prune(model: LeNet5, print_info: bool = False) -> LeNet5:
    """Apply structured pruning to model layers and remove pruning masks

    Args:
        model (LeNet5): Model to prune
        print_info (bool, optional): Print pruning statistics. Defaults to False.

    Returns:
        LeNet5: Pruned model with zeroed weights (structure unchanged)
    """

    prune.ln_structured(model.conv1, name="weight",
                        amount=0.15, n=2, dim=0)
    prune.ln_structured(model.conv2, name="weight",
                        amount=0.15, n=2, dim=0)
    prune.ln_structured(model.fc1, name="weight",
                        amount=0.15, n=2, dim=0)
    prune.ln_structured(model.fc2, name="weight",
                        amount=0.30, n=2, dim=0)
    # removing the weights from the model
    prune.remove(model.conv1, 'weight')
    prune.remove(model.conv2, 'weight')
    prune.remove(model.fc1, 'weight')
    prune.remove(model.fc2, 'weight')
    print("Pruning made permanent - weights removed\n")

    if print_info:
        total_params = 0
        zero_params = 0

        for name, module in model.named_modules():
            if hasattr(module, 'weight'):
                weight = module.weight
                total = weight.numel()
                zeros = (weight == 0).sum().item()

                total_params += total
                zero_params += zeros

                if zeros > 0:
                    print(
                        f"{name:10s}: {total:6d} params, {zeros:6d} zeros ({100*zeros/total:5.1f}%)")

        print(
            f"TOTAL:{total_params:6d} params, {zero_params:6d} zeros ({100*zero_params/total_params:5.1f}%)")

    return model


def model_analyze(model: LeNet5, num_batches: int = MAX_BATCHES):
    """Run validation and print detailed statistics including per-class errors

    Args:
        model (LeNet5): Model to analyze
        num_batches (int, optional): Number of batches to test. Defaults to MAX_BATCHES.

    Returns:
        tuple: Mean accuracy (%), avg time per batch (ms), avg time per sample (ms)
    """

    if num_batches > MAX_BATCHES:
        num_batches = MAX_BATCHES

    model.eval()
    _, dataloader = dataset_load()

    accuracies = []
    total_time = 0.0

    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]

    class_errors = {i: 0 for i in range(10)}
    class_total = {i: 0 for i in range(10)}

    print_interval = max(1, round(math.sqrt(num_batches)))
    print(
        f"Analyzing model on {num_batches} batches ( printing every {print_interval} batches ).\n\n")

    with torch.no_grad():
        for idx, (images, labels) in enumerate(dataloader):

            if idx >= num_batches:
                break

            test_num = idx + 1

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            t_start = time.time()
            outputs = model(images)
            t_stop = time.time() - t_start

            predictions = outputs.argmax(dim=1)
            batch_accuracy = (predictions == labels).float().mean().item()
            t_per_sample = (t_stop / len(outputs)) * 1e3
            t_stop_ms = t_stop * 1e3

            for label, pred in zip(labels.cpu().numpy(), predictions.cpu().numpy()):
                class_total[label] += 1
                if label != pred:
                    class_errors[label] += 1

            accuracies.append(batch_accuracy * 100)
            total_time += t_stop_ms

            if test_num % print_interval == 0 or test_num == num_batches:
                print(f"Batch {test_num}/{num_batches}: Acc={batch_accuracy*100:5.1f}%  Time={t_stop_ms:5.2f}ms  "
                      f"GT={labels.cpu().numpy()}  Pred={predictions.cpu().numpy()}")

    accuracies_np = np.array(accuracies)
    mean_accuracy = np.mean(accuracies_np)
    std_accuracy = np.std(accuracies_np)
    avg_time = total_time / num_batches
    avg_time_per_sample = avg_time / BATCH_SIZE

    class_error_rates = []
    for class_id in range(10):
        if class_total[class_id] > 0:
            error_rate = (class_errors[class_id] / class_total[class_id]) * 100
            class_error_rates.append((class_id, class_names[class_id],
                                     error_rate, class_errors[class_id],
                                     class_total[class_id]))

    print(f"\n\nSTATISTICS\n\n")
    print(f"Device: {DEVICE}")
    print(f"Mean Accuracy:          {mean_accuracy:.2f}%")
    print(f"Std Deviation:          {std_accuracy:.2f}%")
    print(f"Average Time (batch):   {avg_time:.2f} ms")
    print(f"Average Time (sample):  {avg_time_per_sample:.2f} ms")

    print(f"\n\nMOST MISCLASSIFIED CLASSES \n\n")

    for class_id, class_name, error_rate, errors, total in class_error_rates:
        if total > 0:
            print(
                f"{class_id}. {class_name:15s}: {error_rate:5.1f}% error ({errors}/{total} wrong)")
    return mean_accuracy, avg_time, avg_time_per_sample


def model_measure(model: LeNet5, dataloader: DataLoader, num_batches: int = MAX_BATCHES, device="cpu"):
    """Measure model accuracy and inference time on given dataloader

    Args:
        model (LeNet5): Model to test
        dataloader (DataLoader): Validation/test dataloader
        num_batches (int, optional): Number of batches to evaluate. Defaults to MAX_BATCHES.
        device (str, optional): Device for inference. Defaults to "cpu".

    Returns:
        tuple: Accuracy (%), time per batch (ms), time per sample (ms)
    """
    if num_batches > MAX_BATCHES:
        num_batches = MAX_BATCHES

    model.to(device)
    model.eval()
    correct = 0
    total = 0
    start_time = time.time()

    with torch.no_grad():
        for idx, (images, labels) in enumerate(dataloader):
            if idx >= num_batches:
                break

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    t = (time.time() - start_time)  # temp var
    time_per_batch = t * 1000
    time_per_sample = (t / num_batches) * 1000
    return accuracy, time_per_batch, time_per_sample,
