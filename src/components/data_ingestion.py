import os
import sys
from pathlib import Path 
sys.path.append(str(Path(__file__).parent.parent))
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from logger import logging
from exception import CustomException

@dataclass
class dataingestionconfig:
    train_file_path = os.path.join('artifacts' , 'train.csv')
    test_file_path = os.path.join('artifacts' , 'test.csv')
    raw_file_path = os.path.join('artifacts' , 'raw.csv')
    
class dataingestion:
    def __init__(self):
        self.ingestion_config = dataingestionconfig()
        
    def initiate_data_ingestion(self):
        logging.info("data ingestion started")
        try:
            df = pd.read_csv(os.path.join('notebooks/data' , 'wafer_23012020_041211.csv'))
            os.makedirs(os.path.dirname(self.ingestion_config.raw_file_path) , exist_ok = True)
            df.to_csv(self.ingestion_config.raw_file_path , index=False)
            
            logging.info('preprocessing-droping useless columns from the dataframe')
            
            def get_cols_with_zero_std_dev(df: pd.DataFrame):
                """
                Returns a list of columns names who are having zero standard deviation.
                """
                cols_to_drop = []
                num_cols = [col for col in df.columns if df[col].dtype != 'O']  # numerical cols only
                for col in num_cols:
                    if df[col].std() == 0:
                        cols_to_drop.append(col)
                return cols_to_drop

            def get_redundant_cols(df: pd.DataFrame, missing_thresh=.7):
                """
                Returns a list of columns having missing values more than certain thresh.
                """
                cols_missing_ratios = df.isna().sum().div(df.shape[0])
                cols_to_drop = list(cols_missing_ratios[cols_missing_ratios > missing_thresh].index)
                return cols_to_drop  
                        
            cols_to_drop_1 = get_redundant_cols(df, missing_thresh=.7)
            cols_to_drop_2 = get_cols_with_zero_std_dev(df=df)
            cols_to_drop_2.append("Unnamed: 0")
            df.drop(cols_to_drop_1 + cols_to_drop_2 , axis=1, inplace=True)
            logging.info(f"the shape of the dataframe now is : , {df.shape}")
            
            logging.info('Train Test split')
            train_set , test_set = train_test_split(df , test_size=0.33 , random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_file_path , index = False , header = True)            
            test_set.to_csv(self.ingestion_config.test_file_path , index = False , header = True)
            
            logging.info('Ingestion of Data is completed')

            return(
                self.ingestion_config.train_file_path,
                self.ingestion_config.test_file_path
            )
            
        except Exception as e:
            logging.info("error in data ingestion")
            raise CustomException(e,sys)
        