from app.core.config import get_api_settings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
#import csv, json
from typing import Dict, List
import pandas as pd
import numpy as np

settings = get_api_settings()


MODEL_TYPE = settings.model_type

async def launch_model_fitting(model: MODEL_TYPE)->MODEL_TYPE:
    """ preprocess data and fit model variable

    Args:
        model (MODEL_TYPE): model from sklearn library

    Returns:
        MODEL_TYPE: model from sklearn library fitted
    """
    return model