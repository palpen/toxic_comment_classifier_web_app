# Toxic Comment Classification Web App
Flask-based web application for a machine learning model that classifies toxic comments

### How to run locally
```
export FLASK_APP=hello.py
export FLASK_DEBUG=1
python ml_app.py
```

### To do:
- [] Refactor `NBSVM.py` and place in a separate file called `utils.py` inside `flask_api` folder
- [] Create environment yml file
- [] Implement view functions that classifies input comment
- [] Deploy web app
