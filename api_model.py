from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import api_function as model
from pydantic import BaseModel

class textBody(BaseModel):
    text : str

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"error":"False", "message":"Welcome to the API"}

@app.post("/predict", status_code=200)
def predict_semantic(data: textBody):
    try:    
        _text = []
        _text.append(data.text)
        prediction = model.predict_semantic(_text)
        return {"response" : prediction}
    except Exception as e: 
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")