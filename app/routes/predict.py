from fastapi.routing import APIRouter
from fastapi import HTTPException
from app.classes.models import Combined, Match
from typing import List

from app.core.config import get_api_settings
from app.scripts.predict_tools import generate_best_combined
import pickle

settings = get_api_settings()
API_PREDICT_ROUTE = settings.api_predict_route
MODEL_FILE = settings.model_file

PredictRouter = APIRouter()

@PredictRouter.post(API_PREDICT_ROUTE, response_model=Combined)
async def get_best_combined(matchs: List[Match])->Combined:
    """ Generate the combined bet which maximize the gain expectation

    Raises:
        HTTPException: 500 status code if an error is raised during the process

    Returns:
        (DataLine): The Draw generated
    """
    try:
        model = pickle.load(open(MODEL_FILE, "rb"))
        combined = await generate_best_combined(model, matchs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected Error during the draw generation : {e}")
    return combined
