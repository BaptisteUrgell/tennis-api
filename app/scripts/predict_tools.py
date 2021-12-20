from app.core.config import get_api_settings
from app.classes.models import Bet, Combined, Match
from typing import List
import numpy as np
import pandas as pd
#import datetime


settings = get_api_settings()
MODEL_TYPE = settings.model_type


async def generate_best_combined(model: MODEL_TYPE, matchs: List[Match])->Combined:
    bets = [Bet(match=matchs[0],odds=1, prob=1, winner="Monfils")]
    combined = Combined(odds=1, prob=1, exp=1, bets=bets)
    return combined