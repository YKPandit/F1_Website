from fastapi import FastAPI
# from main import getPrediction
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
from SimpleNN import SimpleNN
import torch
import pandas as pd
import numpy as np


class infoRace(BaseModel):
    driver:str
    race:str
    tire_age:int
    previous_lap:float
    weather:str


def predict_time(info):
    prediction = 0
    
    scaler = joblib.load("./Models/scalar.pkl")
    encoder = joblib.load("./Models/encoder.pkl")

    file = open("Models/columns.txt", "r")
    columns = []
    for line in file:
        columns.append(line.strip('\n'))

    other_cols = []
    cols_to_encode = []
    for i in range(5):
        if(i < 2):
            other_cols.append(columns.pop(len(columns)-1))
        else:
            cols_to_encode.append(columns.pop(len(columns)-1))
    
    other_cols.reverse()
    cols_to_encode.reverse()

    input_layer = int(columns.pop(len(columns)-1))

    model = SimpleNN(input_layer, 20, 20, 1)
    model.load_state_dict(torch.load("Models/model.pth", weights_only=False))
    # torch.save(model.state_dict(), "Models/model.pth")

    
    data = pd.DataFrame({
        "Driver":[info.driver],
        "PreviousLap":[info.previous_lap],
        "TireAge" : [info.tire_age],
        "Weather" : [info.weather],
        "Race": [info.race]
    })

    data_encoded = encoder.transform(data[cols_to_encode])
    data_encoded = pd.DataFrame(data_encoded, index=data.index, columns=encoder.get_feature_names_out(cols_to_encode))
    data = data[cols_to_encode + other_cols].join(data_encoded)
    data = data.drop(columns=cols_to_encode)
    data[other_cols] = scaler.transform(data[other_cols])
    data = data.reindex(columns=columns, fill_value=0)
    data_num = data.to_numpy(dtype=np.float32)
    data = torch.from_numpy(data_num)


    print(data)
    model.eval()

    with torch.no_grad():
        predicted = model(data)

        print(f"Predicted lap time: {predicted.item()}")


    return predicted.item()


app = FastAPI()

origins = ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # List of allowed origins
    allow_credentials=True, # Allow cookies, authorization headers etc.
    allow_methods=["*"], # Allow all methods (GET, POST, PUT, DELETE etc.)
    allow_headers=["*"], # Allow all headers
)

@app.post("/")
async def read_root(info:infoRace):
    print(type(info))
    print(info)

    prediction = predict_time(info)
    print(prediction)

    return {"Message": f"Predicted lap: {prediction}"}