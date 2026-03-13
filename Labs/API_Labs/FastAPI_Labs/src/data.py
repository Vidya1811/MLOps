from pydantic import BaseModel
from typing import List


class WineData(BaseModel):

    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float


class WineBatch(BaseModel):
    samples: List[WineData]
