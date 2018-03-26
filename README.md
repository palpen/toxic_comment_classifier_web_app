# Toxic Comment Classification Web App
Flask-based web application for a machine learning model that classifies toxic comments

### How to run locally
```
export FLASK_APP=hello.py
export FLASK_DEBUG=1
python ml_app.py
```

### Very useful references
- [Deploying a Flask app on Heroky](https://github.com/datademofun/heroku-basic-flask)
- [Building a simple Keras deep learning REST API](https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html)
- [Flask API for scikit-learn](https://github.com/amirziai/sklearnflask)

### To do:
- [] Refactor `NBSVM.py` and place in a separate file called `utils.py` inside `flask_api` folder
- [] Deploy web app
- [] Fix view function names
- [X] Create environment yml file (see requirements)
- [X] Implement view functions that classifies input comment

