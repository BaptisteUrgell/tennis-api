from fastapi.routing import APIRouter
from fastapi import HTTPException

from app.core.config import get_api_settings
from app.classes.models import ResponseJson
from app.scripts.model_tools import launch_model_fitting

import pickle

settings = get_api_settings()
API_MODEL_ROUTE = settings.api_model_route
MODEL_FILE = settings.model_file

ModelRouter = APIRouter()


@ModelRouter.post(f"{API_MODEL_ROUTE}/retrain", response_model=ResponseJson)
async def retrain_model() -> ResponseJson:
    """Launch a new fitting of the model with the current dataset 

    Raises:
        HTTPException: 500 status code if an error is raised during the process

    Returns:
        (ResponseJson): Information about the process
    """
    try:
        model = pickle.load(open(MODEL_FILE, "rb"))
        model = await launch_model_fitting(model)
        pickle.dump(model, open(MODEL_FILE, "wb"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected Error during the model retraining : {e}")
    return {"message": "The retraining of the model was done successfully !", "status_code": 200}