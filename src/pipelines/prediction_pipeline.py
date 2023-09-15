import sys
from pathlib import Path 
sys.path.append(str(Path(__file__).parent.parent))
import os
from exception import CustomException
from logger import logging
from utils import load_object
from utils import save_object
import pandas as pd
from flask import request
from dataclasses import dataclass

@dataclass
class PredictionPipelineConfig:
    prediction_output_dirname: str = "predictions"
    prediction_file_name:str =  "predicted_file.csv"
    model_file_path: str = os.path.join('artifacts', "model.pkl" )
    preprocessor_path: str = os.path.join('artifacts', "preprocessor.pkl")
    prediction_file_path:str = os.path.join(prediction_output_dirname,prediction_file_name)

class PredictPipeline:
    def __init__(self, request:request):
        self.request = request
        self.prediction_pipeline_config = PredictionPipelineConfig()
    
    def save_input_files(self)-> str:

        """
            Method Name :   save_input_files
            Description :   This method saves the input file to the prediction artifacts directory. 
            
            Output      :   input dataframe
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """

        try:
            #creating the file
            pred_file_input_dir = "prediction_artifacts"
            os.makedirs(pred_file_input_dir, exist_ok=True)

            input_csv_file = self.request.files['file']
            pred_file_path = os.path.join(pred_file_input_dir, input_csv_file.filename)
            
            
            input_csv_file.save(pred_file_path)
            return pred_file_path
        
        except Exception as e:
            raise CustomException(e,sys)


    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
        
        
    def get_predicted_dataframe(self, input_dataframe_path):

            """
                Method Name :   get_predicted_dataframe
                Description :   this method returns the dataframw with a new column containing predictions

                
                Output      :   predicted dataframe
                On Failure  :   Write an exception log and then raise an exception
                
                Version     :   1.2
                Revisions   :   moved setup to cloud
            """
    
            try:

                prediction_column_name : str = 'Good/Bad'
                input_dataframe: pd.DataFrame = pd.read_csv(input_dataframe_path)
                
                input_dataframe =  input_dataframe.drop(columns="Unnamed: 0") if "Unnamed: 0" in input_dataframe.columns else input_dataframe

                predictions = self.predict(input_dataframe)
                input_dataframe[prediction_column_name] = [pred for pred in predictions]
                target_column_mapping = {0:'bad', 1:'good'}

                input_dataframe[prediction_column_name] = input_dataframe[prediction_column_name].map(target_column_mapping)
                
                os.makedirs( self.prediction_pipeline_config.prediction_output_dirname, exist_ok= True)
                input_dataframe.to_csv(self.prediction_pipeline_config.prediction_file_path, index= False)
                logging.info("predictions completed. ")
                
            except Exception as e:
                logging.info("error in prediction pipeline")
                raise CustomException(e, sys) from e


    def run_pipeline(self):
        try:
            input_csv_path = self.save_input_files()
            self.get_predicted_dataframe(input_csv_path)

            return self.prediction_pipeline_config


        except Exception as e:
            raise CustomException(e,sys)
            