from flask import Flask, request, render_template
from sklearn.externals import joblib
from utils import tokenize  # custom tokenizer required for tfidf model loaded in load_tfidf_model()

app = Flask(__name__)

models_directory = 'models'


def nbsvm_models():
    global tfidf_model
    global logistic_identity_hate_model
    global logistic_insult_model
    global logistic_obscene_model
    global logistic_severe_toxic_model
    global logistic_threat_model
    global logistic_toxic_model

    tfidf_model = joblib.load('{}/tfidf_vectorizer_train.pkl'.format(models_directory))
    logistic_identity_hate_model = joblib.load('models/logistic_identity_hate.pkl')
    logistic_insult_model = joblib.load('models/logistic_insult.pkl')
    logistic_obscene_model = joblib.load('models/logistic_obscene.pkl')
    logistic_severe_toxic_model = joblib.load('models/logistic_severe_toxic.pkl')
    logistic_threat_model = joblib.load('models/logistic_threat.pkl')
    logistic_toxic_model = joblib.load('models/logistic_toxic.pkl')


nbsvm_models()


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

    dict_preds = {}

    dict_preds['pred_identity_hate'] = logistic_identity_hate_model.predict_proba(comment_term_doc)[:, 1][0]
    dict_preds['pred_insult'] = logistic_insult_model.predict_proba(comment_term_doc)[:, 1][0]
    dict_preds['pred_obscene'] = logistic_obscene_model.predict_proba(comment_term_doc)[:, 1][0]
    dict_preds['pred_severe_toxic'] = logistic_severe_toxic_model.predict_proba(comment_term_doc)[:, 1][0]
    dict_preds['pred_threat'] = logistic_threat_model.predict_proba(comment_term_doc)[:, 1][0]
    dict_preds['pred_toxic'] = logistic_toxic_model.predict_proba(comment_term_doc)[:, 1][0]

    for k in dict_preds:
        perc = dict_preds[k] * 100
        dict_preds[k] = "{0:.2f}%".format(perc)

    return render_template('main.html', text=text,
                           pred_identity_hate=dict_preds['pred_identity_hate'],
                           pred_insult=dict_preds['pred_insult'],
                           pred_obscene=dict_preds['pred_obscene'],
                           pred_severe_toxic=dict_preds['pred_severe_toxic'],
                           pred_threat=dict_preds['pred_threat'],
                           pred_toxic=dict_preds['pred_toxic'])


if __name__ == '__main__':

    app.run(debug=True)
