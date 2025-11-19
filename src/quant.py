import torch
from torch import nn
import os
import copy

from model import *
from hparams import *
from dataloader import *

from train import train_model


def main() -> int:

    print("\n\nQuantisizing: float32 -> int8\n\n")

    model_to_quant_path = r"../models/model_base.pth"
    model_quanted_path = r"../models/model_quant_new.pth"

    model_fp32 = model_load(model_to_quant_path)
    model_fp32.to('cpu').eval()
    _, test_loader = dataset_load()

    # quantisize the weights
    model_int8 = torch.quantization.quantize_dynamic(
        copy.deepcopy(model_fp32),
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8
    )

    print("Retraining model for 10 epochs")
    _, criterion, optimizer = model_set()
    extra_epochs = 10

    model_int8.to("cpu")
    DEVICE = "cpu"
    model_int8 = train_model(model_int8, criterion,
                             optimizer, extra_epochs, False, False)
    print("Retraining done.")

    torch.save(model_int8.state_dict(), model_quanted_path)
    print(f"Quantized model saved in {model_quanted_path}")

    # get file sizes
    fp32_size = os.path.getsize(model_to_quant_path) / 1024
    int8_size = os.path.getsize(model_quanted_path) / 1024

    print("Measuring models...")
    fp32_acc, _, fp32_time = model_measure(model_fp32, test_loader)
    int8_acc, _, int8_time = model_measure(model_int8, test_loader)

    print("Model\tsize [kB]\tt_avg [ms]\tacc (current) [%]")
    print(f"base\t{fp32_size:.3f}\t\t{fp32_time:.3f}\t\t{fp32_acc}")
    print(f"quant\t{int8_size:.3f}\t\t{int8_time:.3f}\t\t{int8_acc}")

    return 0


if __name__ == "__main__":
    main()
    # measure_model()
