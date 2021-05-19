import os
import pickle
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from sklearn.pipeline import Pipeline

from src.data_models import InputData, OutputData
from src.validation import validate_input_data

app = FastAPI()
model: Optional[Pipeline] = None


@app.on_event("startup")
def load_model():
    global model
    model_path = 'models/model.pkl'
    with open(model_path, "rb") as f:
        model = pickle.load(f)


@app.get("/predict", response_model=List[OutputData])
def predict(request: InputData):
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model is not loaded"
        )
    data = request.convert_to_pandas()
    if not validate_input_data(data):
        raise HTTPException(
            status_code=400,
            detail="Incorrect input data"
        )
    prediction = model.predict(data)
    return [OutputData(target=x) for x in prediction]


@app.get("/")
def main():
    return "it is entry point of our predictor"


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
