# neuralforge/serve/ab_testing.py
from fastapi import FastAPI, Depends
import random
from pydantic import BaseModel

app = FastAPI()

class Prediction(BaseModel):
    model_version: str
    prediction: float

class ABTest:
    def __init__(self, model_a, model_b, split_ratio=0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.split_ratio = split_ratio

    def get_model(self):
        if random.random() < self.split_ratio:
            return self.model_a, "A"
        else:
            return self.model_b, "B"

ab_test = ABTest(model_a=None, model_b=None)  # Initialize with actual models

@app.post("/predict", response_model=Prediction)
async def predict(data: dict, model_and_version: tuple = Depends(ab_test.get_model)):
    model, version = model_and_version
    prediction = model.predict(data)
    return Prediction(model_version=version, prediction=prediction)

# Usage:
# uvicorn neuralforge.serve.ab_testing:app --reload