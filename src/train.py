from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.model import TinyCNN

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def train(
    epochs: int = 1,
    batch_size: int = 128,
    lr: float = 1e-3,
    limit_batches: int = 200,
) -> str:
    """
    Trains a small CNN on CIFAR-10 and saves weights to artifacts/model_state_dict.pt

    limit_batches keeps training fast for the lab (not aiming for the best accuracy)
    """
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    train_ds = datasets.CIFAR10(root="data", train=True, download=True, transform=tfm)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2
    )

    model = TinyCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_loader):
            if limit_batches and i >= limit_batches:
                break

            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

    Path("artifacts").mkdir(exist_ok=True)
    out_path = "artifacts/model_state_dict.pt"
    torch.save(model.state_dict(), out_path)
    return out_path


if __name__ == "__main__":
    saved = train()
    print("Saved weights:", saved)
