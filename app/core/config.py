from functools import lru_cache
from typing import List
from sklearn.ensemble import RandomForestClassifier

import os

from pydantic import BaseSettings

class APISettings(BaseSettings):

    ########################     Global information    ########################
    
    title: str = "tennis-api"
    contacts: str = "urgellbapt@cy-tech.fr"

    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    
    root_dir = os.path.dirname(os.path.abspath(__file__)) + "/../.."
        
    ########################     Routes    ########################

    api_predict_route: str = "/api/predict"
    api_model_route: str = "/api/model"
        
    ########################     data, model, information ...     ########################
    
    #data_csv: str = root_dir + "/data/EuroMillions_numbers.csv"
    model_file: str = root_dir + "/app/model/model.pkl"
    
    model_type = RandomForestClassifier

    ########################     Other params     ########################

    backend_cors_origins_str: str = ""  # Should be a comma-separated list of origins

    @property
    def backend_cors_origins(self) -> List[str]:
        return [x.strip() for x in self.backend_cors_origins_str.split(",") if x]


@lru_cache()
def get_api_settings() -> APISettings:
    """Init and return the API settings

    Returns:
        APISettings: The settings
    """
    return APISettings()  # reads variables from environment