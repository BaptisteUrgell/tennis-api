from app.core.config import get_api_settings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple
import pandas as pd
import numpy as np
import os, random, re

settings = get_api_settings()


DATA_PATT = settings.data_patt
DATA_ROUTE = settings.data_route
MODEL_TYPE = settings.model_type

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

async def dataframe_drop_missing_value(df: pd.DataFrame, threshold: int = 0):
    """ Delete all row with strictly more missing values than threshold arg

    Args:
        df (pd.DataFrame): Dataframe of all matchs
        threshold (int, optional): number of missing vaues. Defaults to 0.

    Returns:
        pd.DataFrame: Dataframe of all match cleaned of its missing values
    """
    df.dropna(thresh=len(df.columns) - threshold,inplace=True)
    return df

async def dataframe_qual2quan(df: pd.DataFrame)->pd.DataFrame:
    """ Modifie qualitative value into quantitative

    Args:
        df (pd.DataFrame): Dataframe of all matchs

    Returns:
        pd.DataFrame: Dataframe of all matchs with only quantitative values
    """
    df['tourney_name'] = df['tourney_name'].isin(["Roland Garros","Wimbledon","US Open","Australian Open"]).astype(int)
    q2q = ['winner_hand','loser_hand','surface']
    df = df.join(pd.get_dummies(df[q2q[:2]],drop_first=True))
    df = df.join(pd.get_dummies(df[q2q[2]]))
    df.drop(columns=q2q,inplace=True)
    return df

async def add_stat_player(df: pd.DataFrame)->pd.DataFrame:
    """ Calcul number of victory on each surface for all players

    Args:
        df (pd.DataFrame): Dataframe of all matchs

    Returns:
        pd.DataFrame: Dataframe of all matchs plus victory on each surface
    """
    np_df = df.to_numpy()
    new_df = np.zeros((df.shape[0],df.shape[1] + 8), dtype=object)
    index_id = pd.concat([df["winner_id"],df["loser_id"]]).drop_duplicates()
    index_id.reset_index(drop=True,inplace=True)
    index_id = pd.Series(index_id.index,index=index_id.to_numpy())
    surface_victory = np.zeros((index_id.shape[0],4),dtype=int)
    for i in range(np_df.shape[0]):
        surface_victory[index_id[np_df[i,2]]] += np.array(np_df[i,-4:], dtype=int)
        new_df[i] = np.array([*np_df[i], *surface_victory[index_id[np_df[i,2]]], *surface_victory[index_id[np_df[i,5]]]])

    columns = df.columns.to_numpy()
    columns_winner = "winner_" + columns[-4:] 
    columns_loser = "loser_" + columns[-4:]
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

async def create_X(X: pd.DataFrame, index: list)->pd.DataFrame:
    """ Change 50% of data to equal first/second player win

    Args:
        X (pd.DataFrame): Dataframe of all matchs
        index (list): index of X to swap

    Returns:
        pd.DataFrame: Dataframe of all matchs with swapped players
    """
    rename_columns = {}
    for col in list(X.columns):
        rename_columns[col] = col.replace("winner","player0")
        rename_columns[col] = rename_columns[col].replace("loser","player1")
    X = X.rename(columns=rename_columns)
    
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

async def reduc_dim(X):
    for column in X.columns:
        if "player0" in column:
            X[column.replace("player0_","")] = X[column] - X[column.replace("player0","player1")]
            X.drop([column,column.replace("player0","player1")],axis="columns",inplace=True)
    return X

async def preprocess_data()->Tuple[pd.DataFrame, pd.DataFrame]:
    """ Preprocess Data to be used as an input

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: X (features) and y (labels) Dataframes
    """
    df = await importResultMatch()
    df = await dataframe_dim_reduction(df)
    df = await dataframe_drop_missing_value(df)
    df = await dataframe_qual2quan(df)
    df = await add_stat_player(df)
    df.drop(columns="Carpet",inplace=True)
    df.reset_index(inplace=True)
    df.set_index(["tourney_date","winner_id", "loser_id","index"], inplace=True)
    index = await create_index(df)
    X = await create_X(df,index)
    y = await create_y(X,index)
    X = await dataframe_scaler(X)
    X = await reduc_dim(X)
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