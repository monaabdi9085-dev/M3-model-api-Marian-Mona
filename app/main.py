from fastapi import FastAPI, HTTPException
from .schemas import PredictRequest, PredictResponse
from pathlib import Path
from contextlib import asynccontextmanager 
import torch

MODEL_VERSION = "v0-mock"
MODEL_PATH= Path("artifacts/model.torchscript.pt")
Model= None 

@asynccontextmanager 
async def lifespan(app:FastAPI):
    global model, MODEL_VERSION
    if MODEL_PATH.exists():
        model = torch.jit.load(str(MODEL_PATH))
        model.eval()
        MODEL_VERSION= "v1-torchscript"
    else:
         Model = None
    yield # 

app = FastAPI(title="M3 Model API", lifespan=lifespan)


def mock_predict(_image) -> int:
    return 7

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(reciept: PredictRequest):
    try:
        global model 
        
        if model is not None:
            input_tensor = torch.tensor(reciept.image, dtype=torch.float32)
            input_tensor = input_tensor.unsqueeze(0) # batch dimension
            with torch.no_grad():
                output = model(input_tensor)

            probabilities= torch.softmax(output,dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            probabilities_list= probabilities.squeeze().tolist()


        else:
            predicted_class = mock_predict(reciept.image)
        return PredictResponse(
             predicted_cladd = predicted_class,
             model_version=MODEL_VERSION
        )
    except Exception as e:
         raise HTTPException (status_code=500, detail=str(e))
      




 