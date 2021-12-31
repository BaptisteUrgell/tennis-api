import app.scripts.general_tools as gt
from app.core.config import get_api_settings
from app.classes.models import Bet, Combined, Match, Input
from typing import List, Tuple
from itertools import combinations
import pandas as pd

settings = get_api_settings()
MODEL_TYPE = settings.model_type

async def preprocess_data()->pd.DataFrame:
    """ Preprocess Data to be used as an input

    Returns:
        pd.DataFrame: X (features)
    """
    df = await gt.importResultMatch()
    df = await gt.dataframe_dim_reduction(df)
    df = await gt.dataframe_drop_missing_value(df)
    q2q = ['winner_hand','loser_hand','surface']
    df = await gt.dataframe_qual2quan(df, q2q)
    df = await gt.add_stat_player(df)
    df.reset_index(inplace=True)
    df.drop(columns=["tourney_date","winner_id", "loser_id","index"], inplace=True)
    index = await gt.create_index(df)
    X = await gt.create_X(df,index)
    X = await gt.reduc_dim(X)
    data_format = Input(data=X.to_dict())
    X = pd.DataFrame(data_format.data, columns=data_format.columns)
    return X

async def match2Dataframe(match: Match)->pd.DataFrame:
    """ Transform a Match to a Dataframe

    Args:
        match (Match): Match to transform

    Returns:
        pd.DataFrame: Match trasformed
    """
    dict_match = {"tourney_name" : [match.tournament.name],
                  "surface_Grass" : [int(match.tournament.surface == "Grass")],
                  "surface_Carpet" : [int(match.tournament.surface == "Carpet")],
                  "surface_Hard" : [int(match.tournament.surface == "Hard")],
                  "surface_Clay" : [int(match.tournament.surface == "Clay")],
                  "player0_age" : [match.player0.age],
                  "player0_rank_points" : [match.player0.rank_points],
                  "player0_hand_R" : [int(match.player0.hand == "R")],
                  "player0_hand_L" : [int(match.player0.hand == "L")],
                  "player0_Carpet" : [match.player0.carpet],
                  "player0_Grass" : [match.player0.grass],
                  "player0_Hard" : [match.player0.hard],
                  "player0_Clay" : [match.player0.clay],
                  "player1_age" : [match.player1.age],
                  "player1_rank_points" : [match.player1.rank_points],
                  "player1_hand_R" : [int(match.player1.hand == "R")],
                  "player1_hand_L" : [int(match.player1.hand == "L")],
                  "player1_Carpet" : [match.player1.carpet],
                  "player1_Grass" : [match.player1.grass],
                  "player1_Hard" : [match.player1.hard],
                  "player1_Clay" : [match.player1.clay]}
    df = pd.DataFrame(dict_match)
    return df

async def preprocess_data_match(match: Match, data: pd.DataFrame)->pd.DataFrame:
    """ Preprocess a match to match with the features of the model

    Args:
        match (Match): Match to preprocess
        data (pd.DataFrame): Dataframe of all matchs

    Returns:
        pd.DataFrame: Dataframe of the Match in argument
    """
    df = await match2Dataframe(match)
    df['tourney_name'] = df['tourney_name'].isin(["Roland Garros","Wimbledon","US Open","Australian Open"]).astype(int)
    df = await gt.reduc_dim(df)
    data_format = Input(data=df.to_dict())
    df = pd.DataFrame(data_format.data, columns=data_format.columns)
    df.index = [-1]
    df = df.append(data)
    df = await gt.dataframe_scaler(df)
    return df.loc[[-1]]

async def calcul_bet(model: MODEL_TYPE, match: Match, data: pd.DataFrame)->Bet:
    """ Calcul a bet from a Match 

    Args:
        model (MODEL_TYPE): model used to predict the proba to win
        match (Match): Match used to bet on
        data (pd.DataFrame): Dataframe of all matchs preprocessed 

    Returns:
        Bet: Bet of the Match in arg
    """
    bet: Bet

    X = await preprocess_data_match(match, data)
    y_pred_prob = model.predict_proba(X)
    
    if y_pred_prob[0,0] > y_pred_prob[0,1]:
        bet = Bet(winner=match.player0.name, odds=match.odds0, prob=y_pred_prob[0,0], match=match)
    elif y_pred_prob[0,0] < y_pred_prob[0,1]:
        bet = Bet(winner=match.player1.name, odds=match.odds1, prob=y_pred_prob[0,1], match=match)
    elif match.odds0 > match.odds1:
        bet = Bet(winner=match.player0.name, odds=match.odds0, prob=y_pred_prob[0,0], match=match)
    else:
        bet = Bet(winner=match.player1.name, odds=match.odds1, prob=y_pred_prob[0,1], match=match)
        
    return bet

async def calcul_combined(comb: List[Bet])->Combined:
    """ Calcul a Combined stats from a list of Bet

    Args:
        comb (List[Bet]): List of Bet used to calcul the combined

    Returns:
        Combined: Combined calculed
    """
    combined: Combined
    exp: float
    odds: float = 1
    prob: float = 1
    
    for bet in comb:
        odds *= bet.odds
        prob *= bet.prob
    
    exp = (odds * prob) - (1 - prob)
    
    combined = Combined(bets=comb, prob=prob, odds=odds, exp=exp)
    return combined

async def generate_best_combined(model: MODEL_TYPE, matchs: List[Match])->Combined:
    """ Find the best Combined from a list of Match based on the expectation of gain

    Args:
        model (MODEL_TYPE): model used to calcul the proba of win
        matchs (List[Match]): The list of Matchs used

    Returns:
        Combined: Best combined calculed
    """
    combineds: List[Combined] = []
    bets: List[Bet] = []
    best_combined: Combined
    combs: List[Tuple] = []
    
    data = await preprocess_data()
    
    for match in matchs:
        bet = await calcul_bet(model, match, data.copy())
        bets.append(bet)
    
    for n in range(1,len(bets)+1):
        combs = [*combs, *combinations(bets,n)]
    
    for comb in combs:
        combined = await calcul_combined(comb)
        combineds.append(combined)
    
    best_combined = max(combineds, key=lambda x: x.exp)
    return best_combined