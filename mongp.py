from flask import Flask
from flask import jsonify
from flask import request
from flask_pymongo import PyMongo
import datetime
import os
#from dotenv import load_dotenv

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

#load_dotenv()
app.config['MONGO_DBNAME'] = 'logrs'
app.config['MONGO_URI'] = os.getenv('ATLAS_URI')

mongo = PyMongo(app)

df=pd.read_csv("insurance.csv")

@app.after_request
def after_request(response):
  response.headers.set('Access-Control-Allow-Origin', '*')
  response.headers.set('Access-Control-Allow-Headers', 'Content-Type,Authorization')
  response.headers.set('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
  return response

@app.route('/predictions', methods=['GET'])
def get_all_predictions():
  predictions = mongo.db.predictions
  output = []
  for p in predictions.find():
    output.append({
      'creation_time': p['creation_time'],
      'age' : p['age'],
      'sex' : p['sex'],
      'bmi' : p['bmi'],
      'children': p['children'],
      'smoking' : p['smoking'],
      'region' : p['region'],
      'prediction': p['prediction'],
      'id': str(p['_id'])
      })
  return jsonify(output)

#########################return jsonify([{'result' : output}])
@app.route('/pall', methods=['POST'])
def post_new_predictions():
  predictions = mongo.db.predictions
  age = request.json['age']
  bmi = request.json['bmi']
  children = request.json['children']
  sex = request.json['sex']
  smoking = request.json['smoking']
  region = request.json['region']
  #output = {'age' : ['age'], 'bmi' : ['bmi'], 'children' : ['children']}
  
  X=df[['age','bmi','children']].values
  y=df['charges'].values
  regsr=LinearRegression()
  regsr.fit(X,y)
  prediction=regsr.predict(np.asarray([age,bmi,children]).reshape(1,-1))
  pretty_prediction = prediction.tolist()

  post_data = {
    'creation_time': datetime.datetime.utcnow(),
    'age' : age,
    'sex' : sex,
    'bmi' : bmi,
    'children': children,
    'smoking' : smoking,
    'region' : region,
    'prediction': pretty_prediction
  }
  predictions.insert_one(post_data)

  return jsonify({'result' : pretty_prediction})
#########################################

if __name__ == '__main__':
    app.run(debug=True)
