# Fair Stop and Search Assistant
This is a simplification of the final project I did as part of the batch 3 of the Lisbon Data Science Starters Academy. 
The project objective was to develop a classification model to aid police officers in deciding when to search a stopped vehicle. 
The client’s first requirement was for the service to uncover the maximum possible contraband provided that at least half of the conducted searches were successful.  
The second was that it should treat in an equal way the groups belonging to different protected classes, as the police had been under criticism because of its search
policies. 

## Search biases analysis
Before developing the model we were asked to confirm if the suspicions were true and specific groups from protected classes were indeed treated unfairly. 
The results can be found in the notebook: 

**SearchBiasAnalysis.ipynb**  

## Model development
The model development steps, including data cleaning, feature engineering, performance accessment and serialization were conducted in the notebook:

**ModelDevelopment.ipynb**

Auxiliary functions used in the process can be found in:

**aux_functs.py**

## Model deployment 
The model was deployed in heroku as a Flask app. The app can be found in:

**app.py**

## Model deployment results
The model was deployed and receive observations to be classified for a period of one week. The results of the deployment can be found in:

**Deployment Results.ipynb**
