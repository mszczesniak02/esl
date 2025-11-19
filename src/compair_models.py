from model import *
from dataloader import *
from hparams import *
import time
import os
import matplotlib.pyplot as plt
import random

# MODELS FOR COMPARISON

# 91.27% accuracy  - BASE MODEL
MODEL_PATH_BASE = "../models/model_base.pth"
# 91.17% accuracy  - QUANT ONLY MODEL (no improvement from retraining)
MODEL_PATH_QUANT = "../models/model_quant.pth"
# 86.00% accuracy  - PRUNED ONLY MODEL (no improvement from retraining )
MODEL_PATH_PRUNED = "../models/model_pruned.pth"
# MODEL_PATH_PRUNED = "../models/model_pruned_81.50.pth"
# ~85.00% accuracy - PRUNED + QUANT MODEL
MODEL_PATH_PRUNED_QUANT = "../models/model_final.pth"


def model_get_size(model_path: str) -> float:
    """get model size in KB"""
    if not os.path.exists(model_path):
        return None
    size_bytes = os.path.getsize(model_path)
    return size_bytes / 1024


def model_safe_load(model_path, is_quantized=False):
    if not os.path.exists(model_path):
        return None

    try:
        if is_quantized:
            return model_load_quantized(model_path)
        else:
            return model_load(model_path, device='cpu')
    except Exception as e:
        print(
            f"  Warning: Failed to load as {'quantized' if is_quantized else 'float32'}: {e}")
        if is_quantized:
            try:
                return model_load(model_path, device='cpu')
            except:
                return None
        return None


def plot_comparison(images, labels, predictions_dict, model_info):

    label_names = ["T-shirt/top", "Trouser", "Pullover", "Dress",
                   "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    num_models = len(predictions_dict)
    num_samples = 8

    fig, axes = plt.subplots(num_models, num_samples,
                             figsize=(16, 2.5 * num_models),
                             gridspec_kw={'wspace': 0.05, 'hspace': 0.3})

    if num_models == 1:
        axes = axes.reshape(1, -1)

    for model_idx, (model_name, predictions) in enumerate(predictions_dict.items()):
        info = model_info[model_name]

        # title name
        row_title = (f"{model_name}\n"
                     f"Size: {info['size_kb']:.1f} KB | "
                     f"Batch: {info['time_batch_ms']:.2f} ms | "
                     f"Sample: {info['time_sample_ms']:.3f} ms")

        fig.text(0.01, 1 - (model_idx + 0.5) / num_models, row_title,
                 ha='left', va='center', fontsize=10, weight='bold',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        for sample_idx in range(num_samples):
            ax = axes[model_idx, sample_idx]

            image = images[sample_idx].squeeze()
            label = labels[sample_idx].item()
            predicted = predictions[sample_idx].item()

            ax.imshow(image, cmap='gray', aspect='auto')

            is_correct = (label == predicted)
            if is_correct:
                title_text = label_names[label]
                text_color = 'lime'
            else:
                title_text = f"{label_names[label]}\n[{label_names[predicted]}]"
                text_color = 'red'

            ax.text(0.5, 0.95, title_text,
                    color=text_color, fontsize=9, weight='bold',
                    ha='center', va='top', transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            ax.axis('off')

    plt.subplots_adjust(left=0.15, right=0.99, top=0.98, bottom=0.02)
    plt.savefig("../visuals/model_comparison.png",
                dpi=150, bbox_inches='tight')

    plt.show()


def main():
    print("\n\nMODEL COMPARISON: Base vs Quantized vs Pruned vs Pruned+Quantized\n\n")

    _, test_loader = dataset_load()

    batch_idx = random.randint(0, len(test_loader) - 1)
    for idx, (images, labels) in enumerate(test_loader):
        if idx == batch_idx:
            break

    images = images[:8]
    labels = labels[:8]

    predictions_dict = {}
    model_info = {}
    models_to_test = [
        ("Base (float32)", MODEL_PATH_BASE, False),
        ("Quantized (int8)", MODEL_PATH_QUANT, True),
        ("Pruned (float32)", MODEL_PATH_PRUNED, False),
        ("Pruned+Quantized", MODEL_PATH_PRUNED_QUANT, True),
    ]

    for model_name, model_path, is_quantized in models_to_test:

        size_kb = model_get_size(model_path)
        model = model_safe_load(model_path, is_quantized)

        accuracy, time_batch, time_sample = model_measure(
            model, test_loader, device="cpu", num_batches=8)

        model.eval()
        model = model.to('cpu')
        with torch.no_grad():
            outputs = model(images.to('cpu'))
            predictions = outputs.argmax(dim=1)

        # save results
        predictions_dict[model_name] = predictions.cpu()
        model_info[model_name] = {
            "size_kb": size_kb,
            "time_batch_ms": time_batch,
            "time_sample_ms": time_sample
        }

    # show and plot models
    if len(predictions_dict) > 0:
        print("Generating visual comparison...")
        plot_comparison(images.cpu(), labels.cpu(),
                        predictions_dict, model_info)
        print("Comparison saved to: ../visuals/model_comparison.png")

        # compair compression
        if "Base (float32)" in model_info:
            base_size = model_info["Base (float32)"]["size_kb"]
            base_time = model_info["Base (float32)"]["time_batch_ms"]

            print("\n\nCOMPRESSION ANALYSIS:\n")
            for model_name in predictions_dict.keys():
                if model_name == "Base (float32)":
                    continue

                size_ratio = base_size / model_info[model_name]["size_kb"]
                time_diff = (
                    (model_info[model_name]["time_batch_ms"] - base_time) / base_time) * 100

    return 0


if __name__ == "__main__":
    main()
