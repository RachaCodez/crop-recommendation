from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load models
pipeline_crop = joblib.load('pipeline_crop.pkl')
pipeline_yield = joblib.load('pipeline_yield.pkl')
pipeline_fertilizer = joblib.load('pipeline_fertilizer.pkl')
pipeline_pesticide = joblib.load('pipeline_pesticide.pkl')

# Define the data input model using Pydantic
class CropData(BaseModel):
    Soil_Type: str
    Water_Availability: str
    Area_Size: float
    Season: str
    Temperature: float
    Rainfall: str
    Crop_Cycle: str

# Home route with GET method
@app.get("/")
def read_root():
    return {"message": "Welcome to the Crop Prediction API!"}

# Prediction route with POST method
@app.post("/predict/")
def predict(data: CropData):
    df = pd.DataFrame([data.dict()])  # Convert input to a DataFrame

    # Ensure column names match those expected by the model
    df = df.rename(columns={
        'Soil_Type': 'Soil Type',
        'Water_Availability': 'Water Availability',
        'Area_Size': 'Area Size',
        'Season': 'Season',
        'Temperature': 'Temperature',
        'Rainfall': 'Rainfall',
        'Crop_Cycle': 'Crop Cycle'
    })

    try:
        crop_pred = pipeline_crop.predict(df)[0]
        yield_pred = pipeline_yield.predict(df)[0]
        fertilizer_pred = pipeline_fertilizer.predict(df)[0]
        pesticide_pred = pipeline_pesticide.predict(df)[0]

        response = {
            'Crop_Recommendation': crop_pred,
            'Yield_Prediction': yield_pred,
            'Fertilizer_Recommendation': fertilizer_pred,
            'Pesticide_Recommendation': pesticide_pred
        }

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
