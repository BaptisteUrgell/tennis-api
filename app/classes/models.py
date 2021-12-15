from typing import List, Optional
from pydantic import BaseModel  

    
class ResponseJson(BaseModel):
    """ Default Response """
    message: str = "OK"
    status_code: int = 200

class Player(BaseModel):
    """ Player Object containing name """
    name: str  

class Tournament(BaseModel):
    """ Tournament Obect containing name and surface """
    name: str
    surface: str
    
class Match(BaseModel):
    """ Match Object containing players, odds, and tournament """
    player1: Player
    player2: Player
    odds1: float
    odds2: float
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
