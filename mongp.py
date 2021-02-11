from flask import Flask
from flask import jsonify
from flask import request
from flask_pymongo import PyMongo
import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

app.config['MONGO_DBNAME'] = 'logrs'
app.config['MONGO_URI'] = 'mongodb+srv://usertst:usertst@cluster0-mwkmh.mongodb.net/logrs?retryWrites=true&w=majority'

mongo = PyMongo(app)

mongo = PyMongo(app)
df=pd.read_csv("insurance.csv")

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
