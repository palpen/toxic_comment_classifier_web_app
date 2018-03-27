# Toxic Comment Classification Web App
Flask-based web application for a machine learning model that classifies toxic comments

### Data used for training
[Kaggle Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

### How to run locally
```
export FLASK_APP=hello.py
export FLASK_DEBUG=1
python ml_app.py
```

### Very useful references
- [Deploying a Flask app on Heroku](https://github.com/datademofun/heroku-basic-flask) (main reference used for deployment)
- [Building a simple Keras deep learning REST API](https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html)
- [Flask API for scikit-learn](https://github.com/amirziai/sklearnflask)

### Deployment related notes
1. App is deployed using Heroku
2. When you get a `[CRITICAL] WORKER TIMEOUT (pid:26)` when you first start up the application, this may be because it takes a couple of minutes to load one of the models due to its size. Time out occurs after 30 seconds.


### To do:
- [] Create environment file with versions, test all code, redo heroku deployment from scratch (use pip freeze?)
- [] Fix view function names
- [] Add page explaining details of algorithm
- [] Implement deep learning based approach for better performance
- [] Deploy
- [X] Refactor code used for training model and add to repository
- [X] Implement view functions that classifies input comment
