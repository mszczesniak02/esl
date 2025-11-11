# ESL - LeNet5 Fashion MNIST
---
DESC:
---
---
Mateusz Szczęśniak, Wojtek Minior, Kacper Kamiński
---
## Setup

Linux/Mac:
```bash
python -m venv .
source bin/activate
pip install torch torchvision numpy pandas matplotlib scikit-learn tqdm
```

Windows:
```bash
python -m venv .
.\Scripts\activate
pip install torch torchvision numpy pandas matplotlib scikit-learn tqdm
```

## TensorBoard

Training logs:
```bash
tensorboard --logdir=results/final_training_model_log/
```

Hyperparameter tuning logs:
```bash
tensorboard --logdir=results/hparams_tuning_log/
```

## Inference

Run inference with visualization:
```bash
python src/inference.py
```

Change model in `src/hparams.py`:
```python
MODEL_PATH = "models/lenet5_fashion_mnist_0.9127.pth"
```

Available models in `models/` folder.
