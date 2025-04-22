from fastapi import FastAPI
from main import getPrediction
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


class infoRace(BaseModel):
    driver:str
    race:str
    tire_age:str
    previous_lap:str
    weather:str


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

    return {"Message": f"Predicted lap: {getPrediction(info=info).item()}"}