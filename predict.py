
import pickle
from typing import Optional

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = "models/model.bin"

with open(MODEL_PATH, "rb") as f:
    artifact = pickle.load(f)

model = artifact["model"]
threshold = artifact["threshold"]
features = artifact["features"]

app = FastAPI(title="IsItMutual?")


class Participant(BaseModel):
    """Information about one participant and their date."""

    # A. Observable profile
    gender: Optional[float] = None
    age: Optional[float] = None
    age_o: Optional[float] = None
    race: Optional[float] = None
    race_o: Optional[float] = None
    samerace: Optional[float] = None
    field_cd: Optional[float] = None
    goal: Optional[float] = None
    date: Optional[float] = None
    go_out: Optional[float] = None

    # B. Pre-date preferences
    attr1_1: Optional[float] = None
    sinc1_1: Optional[float] = None
    intel1_1: Optional[float] = None
    fun1_1: Optional[float] = None
    amb1_1: Optional[float] = None
    shar1_1: Optional[float] = None

    # C. Participant's post-date impressions
    attr: Optional[float] = None
    sinc: Optional[float] = None
    intel: Optional[float] = None
    fun: Optional[float] = None
    amb: Optional[float] = None
    shar: Optional[float] = None
    like: Optional[float] = None
    prob: Optional[float] = None
    met: Optional[float] = None


@app.get("/")
def health():
    return {
        "status": "ok",
        "model": "IsItMutual?",
        "threshold": threshold
    }


@app.post("/predict")
def predict(participant: Participant):
    row = pd.DataFrame(
        [participant.model_dump()]
    )[features].astype(float)

    probability = float(model.predict_proba(row)[0, 1])

    return {
        "mutual_match_probability": round(probability, 4),
        "mutual_match": bool(probability >= threshold),
        "threshold": threshold
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9696)
