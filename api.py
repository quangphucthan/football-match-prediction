"""
JSON API over model.py.

    uvicorn api:app --reload --port 8000

The model fits on first request (~5s), not at import, so the server binds its
port immediately and `--reload` stays usable.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model import get_model

app = FastAPI(title="Football Match Prediction")

# Vite dev server. Widen this before putting the API anywhere but localhost.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    home: str = Field(min_length=1, max_length=64)
    away: str = Field(min_length=1, max_length=64)
    neutral: bool = False
    tournament: str = Field(default="Friendly", max_length=64)


@app.get("/api/teams")
def teams():
    """Every selectable team, with a colour swatch and its match count."""
    model = get_model()
    return {"teams": model.team_list(), "tournaments": model.tournaments}


@app.post("/api/predict")
def predict(req: PredictRequest):
    """Everything the result screen needs, in one response."""
    try:
        return get_model().predict(req.home, req.away, req.neutral, req.tournament)
    except ValueError as e:
        # model.predict validates team names and tournament against the fitted data.
        raise HTTPException(status_code=422, detail=str(e))
