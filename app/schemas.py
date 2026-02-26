from pydantic import BaseModel, Field
from typing import List, Optional

class PredictRequest(BaseModel): 
    image: List[List[List[float]]] = Field(..., description="3x32x32 array")

class PredictResponse(BaseModel):  
    predicted_class: int 
    probabilities: Optional[List[float]] = None 
    model_version: Optional[str] = None



   