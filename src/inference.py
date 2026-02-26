import torch
import torch.nn.functional as F

CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR10_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def preprocess(image):
    """
    image: nested list with shape 3x32x32
    values: either 0-255 (ints) or 0-1 (floats)
    returns: tensor with shape [1, 3, 32, 32]
    """
    x = torch.tensor(image, dtype=torch.float32)

    # If values are 0-255, scale to 0-1
    if x.max() > 1.5:
        x = x / 255.0

    x = x.unsqueeze(0)  # add batch dimension

    x = (x - CIFAR10_MEAN) / CIFAR10_STD

    return x


def predict_from_torchscript(ts_model, image):
    """
    Runs inference using a TorchScript model.

    Args:
    ts_model: Loaded TorchScript model.
    image: Nested list wiht shape (3, 32, 32).

    Returns:
    pred_class (int): Predicted class index.
    probabilities (lis[float]): Softmax probabilities.
    """

    ts_model.eval()

    x = preprocess(image)

    with torch.no_grad():
        logits = ts_model(x)
        probs = F.softmax(logits, dim=1)[0]
        pred_class = int(torch.argmax(probs).item())

    return pred_class, probs.tolist()
