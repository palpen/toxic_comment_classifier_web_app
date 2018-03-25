from flask import Flask, request, render_template
from sklearn.externals import joblib
from utils import tokenize  # custom tokenizer required for tfidf model loaded in load_tfidf_model()

app = Flask(__name__)

models_directory = 'models'

tfidf_model = None


def load_tfidf_model():
    """
        how to transform a single document
        tfidfvectorizer.transform(['the quick brown fox jumped over the lazy dogs'])
        https://stackoverflow.com/questions/20132070/using-sklearns-tfidfvectorizer-transform
    """
    global tfidf_model

    tfidf_model = joblib.load('{}/tfidf_vectorizer_train.pkl'.format(models_directory))


def load_nbsvm_models():

    global logistic_identity_hate_model
    logistic_identity_hate_model = joblib.load('models/logistic_identity_hate.pkl')

    global logistic_insult_model
    logistic_insult_model = joblib.load('models/logistic_insult.pkl')

    global logistic_obscene_model
    logistic_obscene_model = joblib.load('models/logistic_obscene.pkl')

    global logistic_severe_toxic_model
    logistic_severe_toxic_model = joblib.load('models/logistic_severe_toxic.pkl')

    global logistic_threat_model
    logistic_threat_model = joblib.load('models/logistic_threat.pkl')

    global logistic_toxic_model
    logistic_toxic_model = joblib.load('models/logistic_toxic.pkl')


@app.route('/')
def my_form():
    return render_template('main.html')


@app.route('/', methods=['POST'])
def my_form_post():
    """
        Takes the comment submitted by the user, does some stuff to it,
        then returns it and renders it in the my-form.html template
    """

    text = request.form['text']

    comment_term_doc = tfidf_model.transform([text])

    pred_identity_hate = logistic_identity_hate_model.predict_proba(comment_term_doc)[:, 1]

    return render_template('main.html', text=text, processed_text=pred_identity_hate)


if __name__ == '__main__':

    try:
        load_tfidf_model()
        load_nbsvm_models()
        print("Models loaded")

    except Exception as e:
        print("Model loading failed")
        print(str(e))

    app.run(debug=True)