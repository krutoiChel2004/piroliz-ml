from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np

# Загрузка модели и препроцессора
model = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessing.pkl")

# Определяем входные данные
class PyrolysisInput(BaseModel):
    temperature: float  # Температура (°C)
    time: float         # Время (мин)
    tyre_type: str      # Тип шины (легковая, грузовая, спецтехника)

# Создаём FastAPI приложение
app = FastAPI(title="Pyrolysis Prediction API")

@app.get("/")
def home():
    return {"message": "Pyrolysis Gas Yield Prediction API"}

@app.post("/predict")
def predict(data: PyrolysisInput):
    # Преобразуем входные данные в DataFrame
    input_data = pd.DataFrame({
        "Температура (°C)": [data.temperature],
        "Время (мин)": [data.time],
        "Тип_шины": [data.tyre_type]
    })
    
    # Применяем препроцессинг
    processed_data = preprocessor.transform(input_data)
    
    # Делаем предсказание
    prediction = model.predict(processed_data)
    
    # Возвращаем результат
    return {
        "gas_yield_percentage": round(float(prediction[0]), 2),
        "input_data": data.dict()
    }