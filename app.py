#this is for deployment purpose only
from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import math
from src.logger import logging
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

app=Flask(__name__)


@app.route('/')
def index():
    return render_template('home.html') 


@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))

        )
        pred_df=data.get_data_as_data_frame()

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        results=int(math.floor(min(results[0],100)))
        output= f"the predicted math score is {results}"
        logging.info("predicted the output")
        return render_template('home.html',results=output)
    

if __name__=="__main__":
    app.run(host='127.0.0.1',debug=False)    