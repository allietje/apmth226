# data.py
import numpy as np
from torchvision import datasets, transforms


# to load MNIST if it's not already in data/MNIST
def load_mnist(
    flatten: bool = True,
    normalize: bool = True,
):
    """
    Load MNIST using torchvision and return NumPy arrays.

    Returns:
        (X_train, y_train), (X_test, y_test)
        X_*: float32, shape (N, 784) if flatten else (N, 28, 28)
        y_*: one-hot float32, shape (N, 10)
    """
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(
        root='./data/MNIST', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data/MNIST', train=False, download=True, transform=transform
    )

    X_train = train_dataset.data.numpy().astype(np.float32)
    X_test = test_dataset.data.numpy().astype(np.float32)

    if normalize:
        X_train /= 255.0
        X_test /= 255.0

    if flatten:
        X_train = X_train.reshape(-1, 28 * 28)
        X_test = X_test.reshape(-1, 28 * 28)

    num_classes = 10
    y_train = np.eye(num_classes, dtype=np.float32)[train_dataset.targets.numpy()]
    y_test = np.eye(num_classes, dtype=np.float32)[test_dataset.targets.numpy()]

    return (X_train, y_train), (X_test, y_test)
