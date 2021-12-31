from fastapi.routing import APIRouter
from fastapi import HTTPException
from typing import List
from app.core.config import get_api_settings
from app.classes.models import Player, ResponseJson
from app.scripts.model_tools import retrieve_players, launch_model_fitting, preprocess_players
from sklearn.ensemble import RandomForestClassifier


import pickle

settings = get_api_settings()
API_MODEL_ROUTE = settings.api_model_route
MODEL_FILE = settings.model_file

ModelRouter = APIRouter()


@ModelRouter.post(f"{API_MODEL_ROUTE}/train", response_model=ResponseJson)
async def train_model() -> ResponseJson:
    """Launch a new fitting of the model with the current dataset 

    Raises:
        HTTPException: 500 status code if an error is raised during the process

    Returns:
        (ResponseJson): Information about the process
    """
    try:
        model = RandomForestClassifier(random_state=0)
        model = await launch_model_fitting(model)
        pickle.dump(model, open(MODEL_FILE, "wb"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected Error during the model retraining : {e}")
    return {"message": "The retraining of the model was done successfully !", "status_code": 200}

@ModelRouter.get(f"{API_MODEL_ROUTE}/players", response_model=List[Player])
async def players_model() -> List[Player]:
    """ List all players in the dataset

    Raises:
        HTTPException: 500 status code if an error is raised during the process

    Returns:
        List[Player]: List of players
    """
    try:
        df = await preprocess_players()
        players = await retrieve_players(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected Error during the players information loading : {e}")
    return players