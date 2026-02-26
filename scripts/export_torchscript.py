from pathlib import Path
import torch

from src.model import TinyCNN


def export_torchscript() -> str:
    Path("artifacts").mkdir(exist_ok=True)

    weights_path = Path("artifacts/model_state_dict.pt")
    if not weights_path.exists():
        raise FileNotFoundError(
            "Model weights not found at 'artifacts/model_state_dict.pt'."
            "Please run: uv run python -m src.train before exporting."
        )

    model = TinyCNN()
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    example_input = torch.randn(1, 3, 32, 32)
    ts_model = torch.jit.trace(model, example_input)

    out_path = Path("artifacts/model.torchscript.pt")
    ts_model.save(out_path.as_posix())
    return out_path.as_posix()


if __name__ == "__main__":
    saved = export_torchscript()
    print("Saved TorchScript:", saved)
