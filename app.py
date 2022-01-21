import os
import json
import pickle
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
     SqliteDatabase, PostgresqlDatabase, Model, IntegerField,CharField,
     FloatField, BooleanField ,TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict
from sklearn.externals import joblib
from aux_functs import *

from sklearn.base import TransformerMixin, BaseEstimator



##############################
# Begin database stuff

if 'DATABASE_URL' in os.environ:
    db_url = os.environ['DATABASE_URL']
    dbname = db_url.split('@')[1].split('/')[1]
    user = db_url.split('@')[0].split(':')[1].lstrip('//')
    password = db_url.split('@')[0].split(':')[2]
    host = db_url.split('@')[1].split('/')[0].split(':')[0]
    port = db_url.split('@')[1].split('/')[0].split(':')[1]
    DB = PostgresqlDatabase(
        dbname,
        user=user,
        password=password,
        host=host,
        port=port,
    )
else:
    DB = SqliteDatabase('predictions.db')


class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    pred = BooleanField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)

# End database stuff
###############################

###############################
# Unpickle the previously-trained model_to_dict

with open('columns.json') as fh:
    columns = json.load(fh)

pipeline = joblib.load('pipeline.pickle')

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)

# End model un-pickling
#################################


#################################
# Begin webserver stuff

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    obs_dict = request.get_json(force=True)
    _id = obs_dict['id']
    observation = obs_dict['observation']
    obs = pd.DataFrame([observation], columns = columns).astype(dtypes)
    pred = bool(pipeline.predict(obs)[0])
    response = {'ContrabandIndicator': pred}
    p = Prediction(
        observation_id=_id,
        pred=pred,
        observation=request.data,
    )
    try:
        p.save()
    except IntegrityError:
        error_msg = "ERROR: Observation ID: '{}' already exists".format(_id)
        response['error'] = error_msg
        print(error_msg)
        DB.rollback()
    return jsonify(response)


@app.route('/update',methods=['POST'])
def update():
    obs = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id == obs['id'])
        p.true_class = obs['true_class']
        p.save()
        return jsonify(model_to_dict(p))
    except Prediction.DoesNotExist:
        error_msg = 'Observatiob ID: "{}" does not exist'.format(obs['id'])
        return jsonify({"error": error_msg})

# End webserver stuff
#####################################

if __name__ == "__main__":
    app.run(debug=True)
