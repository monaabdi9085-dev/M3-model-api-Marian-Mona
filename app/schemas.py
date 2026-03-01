
from pydantic import BaseModel, Field
from typing import List, Optional

class PredictRequest(BaseModel): 
    image: List[List[List[float]]] = Field(
        ...,
        description="Image tensor with 32x32 spatial dimensions (supports multiple channel formats, e.g., 1x32x32 or 3x32x32)"
    )

class PredictResponse(BaseModel):  
    predicted_class: int 
    probabilities: Optional[List[float]] = None 
    model_version: Optional[str] = None

    