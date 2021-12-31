#from app.classes.models import 
from pandas.core.arrays.sparse import dtype
from app.core.config import get_api_settings
from sklearn.preprocessing import MinMaxScaler
from typing import List
import pandas as pd
import numpy as np
import os, random, re

settings = get_api_settings()

DATA_PATT = settings.data_patt
DATA_ROUTE = settings.data_route

async def importResultMatch()->pd.DataFrame:
    """ import all csv of single player male as pd.Dataframe

    Returns:
        pd.DataFrame: Dataframe of all matchs
    """
    large_df: pd.DataFrame
    
    small_dfs = []
    patt_comp = re.compile(DATA_PATT)
    for file in os.listdir(DATA_ROUTE):
        if patt_comp.match(file):
            small_dfs.append(pd.read_csv(DATA_ROUTE+file))
    large_df = pd.concat(small_dfs, ignore_index=True)
    return large_df

async def dataframe_dim_reduction(df: pd.DataFrame)->pd.DataFrame:
    """ Keep only useful features

    Args:
        df (pd.DataFrame): Dataframe of all matchs

    Returns:
        pd.DataFrame: Dataframe of all matchs with only usful features
    """
    df = pd.DataFrame(df[['tourney_date','tourney_name','surface',
                          'winner_id','winner_hand','winner_age','winner_rank_points',
                          'loser_id','loser_hand','loser_age','loser_rank_points']])
    df["tourney_date"] = pd.to_datetime(df['tourney_date'].astype(str), format='%Y%m%d')
    return df

async def dataframe_drop_missing_value(df: pd.DataFrame):
    """ Delete all row with strictly more missing values than threshold var

    Args:
        df (pd.DataFrame): Dataframe of all matchs

    Returns:
        pd.DataFrame: Dataframe of all match cleaned of its missing values
    """
    threshold = 0
    df.dropna(thresh=len(df.columns) - threshold,inplace=True)
    return df

async def dataframe_qual2quan(df: pd.DataFrame, q2q: List[str])->pd.DataFrame:
    """ Modifie qualitative value into quantitative

    Args:
        df (pd.DataFrame): Dataframe of all matchs
        q2q (List[str]): List of columns names to change

    Returns:
        pd.DataFrame: Dataframe of all matchs with only quantitative values
    """
    df['tourney_name'] = df['tourney_name'].isin(["Roland Garros","Wimbledon","US Open","Australian Open"]).astype(int)
    df = df.join(pd.get_dummies(df[q2q]))
    df.drop(columns=q2q,inplace=True)
    return df

async def add_stat_player(df: pd.DataFrame)->pd.DataFrame:
    """ Calcul number of victory on each surface for all players

    Args:
        df (pd.DataFrame): Dataframe of all matchs

    Returns:
        pd.DataFrame: Dataframe of all matchs plus victory on each surface
    """
    df.sort_values("tourney_date",inplace=True)
    np_df = df.to_numpy()
    col = df.columns.tolist()
    winner_id_index = col.index("winner_id")
    loser_id_index = col.index("loser_id")
    surface_index = np.array([col.index("surface_Carpet"), col.index("surface_Hard"), col.index("surface_Clay"), col.index("surface_Grass")], dtype=int)
    
    new_df = np.zeros((df.shape[0],df.shape[1] + 8), dtype=object)
    index_id = pd.concat([df["winner_id"],df["loser_id"]]).drop_duplicates()
    index_id.reset_index(drop=True,inplace=True)
    index_id = pd.Series(index_id.index,index=index_id.to_numpy())
    surface_victory = np.zeros((index_id.shape[0],4),dtype=int)
    for i in range(np_df.shape[0]):
        surface_victory[index_id[np_df[i,winner_id_index]]] += np.array(np_df[i,surface_index], dtype=int)
        new_df[i] = np.array([*np_df[i], *surface_victory[index_id[np_df[i,winner_id_index]]], *surface_victory[index_id[np_df[i,loser_id_index]]]])

    columns = df.columns.to_numpy()
    winner = lambda c: str(c).replace("surface","winner")
    loser = lambda c: str(c).replace("surface","loser")
    columns_winner = np.vectorize(winner,otypes=[str])(columns[surface_index])
    columns_loser = np.vectorize(loser,otypes=[str])(columns[surface_index])
    columns = pd.Index([*columns, *columns_winner, *columns_loser])
    df = pd.DataFrame(new_df, columns=columns)
    return df

async def create_index(df: pd.DataFrame)->list:
    """ Create a random list of index from Dataframe

    Args:
        df (pd.DataFrame): Dataframe of all matchs

    Returns:
        List: list of Dataframe index
    """
    index = random.sample(list(df.index), len(df.index) // 2)
    return index

def swap2elements(l: List[str], e1: str, e2: str)->List[str]:
    """ Swap 2 elements in the given list

    Args:
        l (List[str]): List of string
        e1 (str): first element to swap
        e2 (str): second element to swap

    Returns:
        List[str]: List of string with elements swaped 
    """
    i1, i2 = l.index(e1), l.index(e2)
    l[i2], l[i1] = l[i1], l[i2]
    return l

async def rename_columns(df: pd.DataFrame)->pd.DataFrame:
    """ rename columns of the dataframe

    Args:
        df (pd.DataFrame): dataframe of all matchs

    Returns:
        pd.DataFrame: Dataframe of all matchs with columns changed
    """
    new_columns = {}
    for col in list(df.columns):
        new_columns[col] = col.replace("winner","player0")
        new_columns[col] = new_columns[col].replace("loser","player1")
    df = df.rename(columns=new_columns)
    return df

async def create_X(X: pd.DataFrame, index: list)->pd.DataFrame:
    """ Change 50% of data to equal first/second player win

    Args:
        X (pd.DataFrame): Dataframe of all matchs
        index (list): index of X to swap

    Returns:
        pd.DataFrame: Dataframe of all matchs with swapped players
    """
    X = await rename_columns(X)
    
    new_columns = list(X.columns)
    swap_list = []
    for column in new_columns:
        if "player0" in column:
            swap_list.append((column,column.replace("player0","player1")))
    
    for col1, col2 in swap_list:
        new_columns = swap2elements(new_columns, col1, col2)
    
    df_temp = X.loc[index].reindex(columns=new_columns)
    df_temp.columns = X.columns
    X.loc[index] = df_temp
    return X

async def create_y(X: pd.DataFrame, index: list)->pd.DataFrame:
    """ Create the y vector for the Dataframe X

    Args:
        X (pd.DataFrame): Dataframe of all matchs
        index (list): index of X to swap

    Returns:
        pd.DataFrame: y vector
    """
    y = pd.DataFrame(data=np.zeros(len(X.index)),index=X.index,columns=['player'])
    y.loc[index,'player'] = 1
    return y

async def dataframe_scaler(df: pd.DataFrame)->pd.DataFrame:
    """ Scale the Dataframe between 0 and 1

    Args:
        df (pd.DataFrame): Dataframe of all matchs

    Returns:
        pd.DataFrame: Dataframe of all matchs scalled
    """
    for column in df.columns:
        df[column] = df[column].astype('float')
    df = pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns, index=df.index)
    return df

async def reduc_dim(df: pd.DataFrame)->pd.DataFrame:
    """ Reduce dimention of the features by subtract players stats for a match

    Args:
        df (pd.DataFrame): Dataframe of all matchs

    Returns:
        pd.DataFrame: Dataframe of all matchs with dimention reducted
    """
    for column in df.columns:
        if "player0" in column:
            df[column.replace("player0_","")] = df[column] - df[column.replace("player0","player1")]
            df.drop([column,column.replace("player0","player1")],axis="columns",inplace=True)
    return df