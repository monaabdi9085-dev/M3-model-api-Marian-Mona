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
    x = torch.tensor(image, dtype=torch.float32)

    if x.ndim == 3 and x.shape[2] == 3:
        x = x.permute(2, 0, 1)

    if x.ndim == 2:
        x = x.unsqueeze(0).repeat(3, 1, 1)

    if not (x.ndim == 3 and x.shape[0] == 3):
        raise HTTPException(
            status_code=422,
            detail=f"Bad image tensor shape {tuple(x.shape)}. Expected [H,W,3] or [3,H,W] or [H,W]."
        )

    x = x.unsqueeze(0)

    if x.numel() > 0 and x.max() > 1.5:
        x = x / 255.0

    return x


@app.post("/predict", response_model=PredictResponse)
def predict(receipt: PredictRequest):
    try:
        # Validate spatial dimensions (H=W=32)
        if len(receipt.image[0]) != 32 or len(receipt.image[0][0]) != 32:
            raise HTTPException(
                status_code=422,
                detail="Input image must have spatial dimensions 32x32."
            )

        if model is not None:
            input_tensor = to_model_tensor(receipt.image)

            with torch.no_grad():
                output = model(input_tensor)

            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            probabilities_list = probabilities.squeeze().tolist()
        else:
            predicted_class = mock_predict(receipt.image)
            probabilities_list = None

        return PredictResponse(
            predicted_class=predicted_class,
            probabilities=probabilities_list,
            model_version=MODEL_VERSION
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))