import os
import sys
import numpy as np 
from pathlib import Path 
sys.path.append(str(Path(__file__).parent.parent))
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from logger import logging
from exception import CustomException
from components import data_ingestion
from components import data_transformation
from components import model_trainer

obj = data_ingestion.dataingestion()
train_path , test_path = obj.initiate_data_ingestion()
print(train_path , test_path)

data_transformation = data_transformation.DataTransformation()
    
train_arr , test_arr = data_transformation.initiate_data_transformation(train_data_path=train_path , test_data_path=test_path)
    
model_trainer = model_trainer.ModelTrainer()
model_trainer.initiate_model_training(train_array=train_arr , test_array=test_arr)

