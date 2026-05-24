from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from model_utils import load_models, predict_yield

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load all models + dataset once at startup
models = load_models()


class PredictRequest(BaseModel):
    year: int
    country: str
    crop: str
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def index():
    return FileResponse("index.html")


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        # predict_yield returns a dict: {"yield": float, "note": str}
        result = predict_yield(
            year=req.year,
            country=req.country,
            crop=req.crop,
            models=models,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))