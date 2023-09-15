from flask import Flask,request,render_template,jsonify, send_file
from src.pipelines.prediction_pipeline import PredictPipeline
import pandas as pd
import sys
from pathlib import Path 
sys.path.append(str(Path(__file__).parent.parent))
from logger import logging

application=Flask(__name__)

app=application

@app.route('/')
def home_page():
    return render_template('upload_file.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        predict_pipeline=PredictPipeline(request)
        #now we are running this run pipeline method
        prediction_file_detail = predict_pipeline.run_pipeline()

        logging.info("prediction completed. Downloading prediction file.")
        return send_file(prediction_file_detail.prediction_file_path,
                         download_name= prediction_file_detail.prediction_file_name,
                        as_attachment= True)
        
        # results=round(pred[0],2)

        # return render_template('form.html',final_result=results)
    

if __name__=="__main__":
    app.run(host='0.0.0.0', port = 8888 , debug=True)
    
    
    