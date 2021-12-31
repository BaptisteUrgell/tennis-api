from app.classes.models import Player, Input
from app.core.config import get_api_settings
from sklearn.model_selection import train_test_split
import app.scripts.general_tools as gt
from typing import List, Tuple
import pandas as pd
import numpy as np


settings = get_api_settings()


MODEL_TYPE = settings.model_type


async def preprocess_data()->Tuple[pd.DataFrame, pd.DataFrame]:
    """ Preprocess Data to be used as an input

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: X (features) and y (labels) Dataframes
    """
    df = await gt.importResultMatch()
    df = await gt.dataframe_dim_reduction(df)
    df = await gt.dataframe_drop_missing_value(df)
    q2q = ['winner_hand','loser_hand','surface']
    df = await gt.dataframe_qual2quan(df, q2q)
    df = await gt.add_stat_player(df)
    df.reset_index(inplace=True)
    df.set_index(["tourney_date","winner_id", "loser_id","index"], inplace=True)
    index = await gt.create_index(df)
    X = await gt.create_X(df,index)
    y = await gt.create_y(X,index)
    X = await gt.reduc_dim(X)
    X = await gt.dataframe_scaler(X)
    data_format = Input(data=X.to_dict())
    X = pd.DataFrame(data_format.data, columns=data_format.columns)
    return X, y

async def launch_model_fitting(model: MODEL_TYPE)->MODEL_TYPE:
    """ preprocess data and fit the model

    Args:
        model (MODEL_TYPE): model from sklearn library

    Returns:
        MODEL_TYPE: model from sklearn library fitted
    """
    X, y = await preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    model.fit(X_train,y_train)
    return model

async def retrieve_players(df: pd.DataFrame)->List[Player]:
    """ List all players in the dataset df

    Args:
        df (pd.DataFrame): dataset of all matchs

    Returns:
        List[Player]: List of players
    """
    df = df.to_dict()
    df = pd.DataFrame(df, columns=["winner_name","winner_id","winner_age","winner_rank_points","winner_hand","winner_Carpet","winner_Grass","winner_Hard","winner_Clay",
                                   "loser_name","loser_id","loser_age","loser_rank_points","loser_hand","loser_Carpet","loser_Grass","loser_Hard","loser_Clay"])
    np_df = df.to_numpy()
    index_id = pd.concat([df["winner_id"],df["loser_id"]]).drop_duplicates()
    index_id.reset_index(drop=True,inplace=True)
    index_id = pd.Series(index_id.index,index=index_id.to_numpy())
    np_players = np.full(index_id.shape[0],Player(name="",id=0, age=-1,rank_points=0,hand="",carpet=0,grass=0,hard=0,clay=0),dtype=Player)
    for i in range(np_df.shape[0]):
        if np_players[index_id[np_df[i,1]]].age < np_df[i,2]:
            np_players[index_id[np_df[i,1]]] = Player(name=np_df[i,0], id=np_df[i,1], age=np_df[i,2],rank_points=np_df[i,3],hand=np_df[i,4],carpet=np_df[i,5],grass=np_df[i,6],hard=np_df[i,7],clay=np_df[i,8])
        if np_players[index_id[np_df[i,10]]].age < np_df[i,11]:
            np_players[index_id[np_df[i,10]]] = Player(name=np_df[i,9], id=np_df[i,10], age=np_df[i,11],rank_points=np_df[i,12],hand=np_df[i,13],carpet=np_df[i,14],grass=np_df[i,15],hard=np_df[i,16],clay=np_df[i,17])
    players = np_players.tolist()
    return players

async def dataframe_dim_reduction(df: pd.DataFrame)->pd.DataFrame:
    """ Keep only useful features

    Args:
        df (pd.DataFrame): Dataframe of all matchs

    Returns:
        pd.DataFrame: Dataframe of all matchs with only usful features
    """
    df = pd.DataFrame(df[['tourney_date','tourney_name','surface',"winner_name","loser_name",
                          'winner_id','winner_hand','winner_age','winner_rank_points',
                          'loser_id','loser_hand','loser_age','loser_rank_points']])
    df["tourney_date"] = pd.to_datetime(df['tourney_date'].astype(str), format='%Y%m%d')
    return df

async def preprocess_players()->pd.DataFrame:
    """ Preprocess data to be used to enumerate players

    Returns:
        pd.DataFrame: Dataframe of all matchs
    """
    df = await gt.importResultMatch()
    df = await dataframe_dim_reduction(df)
    df = await gt.dataframe_drop_missing_value(df)
    q2q = ['surface']
    df = await gt.dataframe_qual2quan(df, q2q)
    df = await gt.add_stat_player(df)
    return df