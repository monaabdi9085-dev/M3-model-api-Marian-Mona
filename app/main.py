from fastapi import FastAPI, HTTPException
from .schemas import PredictRequest, PredictResponse
from pathlib import Path
from contextlib import asynccontextmanager
import torch

MODEL_VERSION = "v0-mock"
MODEL_PATH = Path("artifacts/model.torchscript.pt")
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, MODEL_VERSION
    if MODEL_PATH.exists():
        model = torch.jit.load(str(MODEL_PATH))
        model.eval()
        MODEL_VERSION = "v1-torchscript"
    else:
        model = None
    yield

app = FastAPI(title="M3 Model API", lifespan=lifespan)


def mock_predict(_image) -> int:
    return 7


@app.get("/health")
def health():
    return {"status": "ok"}


def to_model_tensor(image) -> torch.Tensor:
    """
    Accepts ONLY HWC image with shape [32][32][3].
    Converts to tensor shape [1, 3, 32, 32] for the model.
    """
    x = torch.tensor(image, dtype=torch.float32)

    # Validate HWC shape
    if not (x.ndim == 3 and x.shape[0] == 32 and x.shape[1] == 32 and x.shape[2] == 3):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid image shape {tuple(x.shape)}. "
                "Expected HWC format with shape (32, 32, 3)."
            ),
        )

    # Normalize 0-255 -> 0-1 if needed
    if x.numel() > 0 and x.max() > 1.5:
        x = x / 255.0

    # HWC -> CHW and add batch dim
    x = x.permute(2, 0, 1).unsqueeze(0)  # [1,3,32,32]
    return x

@app.post("/predict", response_model=PredictResponse)
def predict(receipt: PredictRequest) -> PredictResponse:
    try:
        if model is None:
            predicted_class = mock_predict(receipt.image)
            return PredictResponse(
                predicted_class=predicted_class,
                probabilities=None,
                model_version=MODEL_VERSION,
            )

        input_tensor = to_model_tensor(receipt.image)  # -> [1, 3, 32, 32]

        with torch.no_grad():
            output = model(input_tensor)  # -> [1, 10]

        probabilities = torch.softmax(output, dim=1).squeeze(0)  # -> [10]
        predicted_class = int(torch.argmax(probabilities).item())

        return PredictResponse(
            predicted_class=predicted_class,
            probabilities=[float(p) for p in probabilities.tolist()],
            model_version=MODEL_VERSION,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))