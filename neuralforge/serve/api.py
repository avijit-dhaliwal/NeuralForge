from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import numpy as np
from PIL import Image
import io

app = FastAPI()

class PredictionInput(BaseModel):
    data: List[float]

class PredictionOutput(BaseModel):
    prediction: float

model = None  # Load your trained model here

@app.post("/predict", response_model=PredictionOutput)
async def predict(input: PredictionInput):
    data = np.array(input.data).reshape(1, -1)
    prediction = model.predict(data)
    return PredictionOutput(prediction=float(prediction[0]))

@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image = image.resize((224, 224))  # Resize to match model input size
    image_array = np.array(image) / 255.0
    prediction = model.predict(image_array[np.newaxis, ...])
    return {"prediction": float(prediction[0])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)