from typing import List, Optional
from pydantic import BaseModel  

    
class ResponseJson(BaseModel):
    """ Default Response """
    message: str = "OK"
    status_code: int = 200

class Player(BaseModel):
    """ Player Object containing name """
    name: str
    id: int
    age: float
    rank_points: int
    hand: str
    carpet: int
    grass: int
    hard: int
    clay: int
    

class Tournament(BaseModel):
    """ Tournament Obect containing name and surface """
    name: str
    surface: str
    
class Match(BaseModel):
    """ Match Object containing players, odds, and tournament """
    player0: Player
    player1: Player
    odds0: float
    odds1: float
    tournament: Tournament
    
class Bet(BaseModel):
    """ Bet Object containing winner, probability of succes, odds and the associated match """
    winner: str
    prob: float
    odds: float
    match: Match   
   
class Combined(BaseModel):
    """ Combined Object containing list of Bet, gain expectation, probability of succes, odds """
    bets: List[Bet]
    exp: float
    prob: float
    odds: float

class Input(BaseModel):
    """ Format for features in the model """
    data: dict
    columns: List[str] = ["tourney_name","surface_Clay","surface_Grass","surface_Hard","age","rank_points","hand_R","hand_L","Carpet","Grass","Hard","Clay"]