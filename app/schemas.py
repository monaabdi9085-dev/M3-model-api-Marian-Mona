from pydantic import BaseModel, Field
from typing import List, Optional

class PredictRequest(BaseModel):
    image: List[List[List[float]]] = Field(
        ...,
        description="Image in HWC format with shape 32x32x3 (values 0-255 or 0-1)."
    )

class PredictResponse(BaseModel):
    predicted_class: int
    probabilities: Optional[List[float]] = None
    model_version: Optional[str] = None



