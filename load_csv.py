import pandas as pd

def ft_load_csv(path: str) -> pd.DataFrame:
    """
    Extract data from csv file
    
    :param path: filepath
    """
    
    df = pd.read_csv(path)
    return df