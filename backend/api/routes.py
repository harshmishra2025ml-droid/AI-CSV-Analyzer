from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
import io

from backend.services.profiler import profile_dataframe
from backend.services.ml_analyzer import ml_analyze
from backend.services.ai_analyzer import ai_analyze

router = APIRouter()

@router.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a CSV file.")
    raw = await file.read()
    if len(raw) > 100 * 1024 * 1024:
        raise HTTPException(413, "File is larger than 100 MB.")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(400, f"Could not read CSV: {exc}")
    profile = profile_dataframe(df)
    ml = ml_analyze(df)
    return {"filename": file.filename, "profile": profile, "ml": ml}

@router.post("/ai")
async def ai(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a CSV file.")
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(400, f"Could not read CSV: {exc}")
    profile = profile_dataframe(df)
    ml = ml_analyze(df)
    result = ai_analyze(profile, ml)
    return result
