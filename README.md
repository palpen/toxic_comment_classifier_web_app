# Toxic Comment Classification Web App
Flask-based web application for a machine learning model that classifies toxic comments.

You can test it here (http://pspenano.pythonanywhere.com)

I use a modified implementation of the NBSVM algorithm (using logistic regression instead of SVM) to train on a corpus of comments from Wikipedia edits (see below for data source). The model is from [this](https://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf) paper and the implementation of the algorithm is by Jeremy Howard (found [here](https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline)).


### Data used for training
[Kaggle Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

### How to run locally
```
export FLASK_APP=hello.py
export FLASK_DEBUG=1
python ml_app.py
```

### Very useful references
- [Deploying a Flask app on Heroku](https://github.com/datademofun/heroku-basic-flask)
- [Building a simple Keras deep learning REST API](https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html)
- [Flask API for scikit-learn](https://github.com/amirziai/sklearnflask)

### Deployment related notes
1. App is deployed on pythonanywhere.com
2. Heroku has a hard limit of 512mb RAM for both the free and hobby tiers. This app consumes roughly 500mb per worker, which is not ideal. Pythonanywhere offers significantly more ram at 3gb. Also, the hobby tier allows you to add more storage at 25 cents per gb.
3. The app needs `sklearn`, `numpy`, and `scipy` (see requirements.txt) to be able to load the trained models. When I installed the requirements, I hit the 512mb storage limit in the free tier immediately, which means you need at minimum roughly 1gb of storage.
3. the `-w 2` option in `Procfile` limits the number of workers to 2 to minimize memory accumulation due to leak (each dyno only has 512mb of ram)

### To do:
- [] Fix view function names
- [] Add page explaining details of algorithm
- [] Add database to backend to save submitted comments
- [] Implement deep learning based approach for better performance
- [X] Create environment file with versions, test all code, redo heroku 
- [X] Deploy
- [X] Refactor code used for training model and add to repository
- [X] Implement view functions that classifies input comment
