from flask import Flask, request, render_template
from utils import tokenize  # custom tokenizer required for tfidf model loaded in load_tfidf_model()
import pickle
import os

app = Flask(__name__)
basepath = os.path.abspath(".")

# models_directory = 'models'


# @app.before_first_request
# def nbsvm_models():
#     # from utils import tokenize

#     global tfidf_model
#     global logistic_identity_hate_model
#     global logistic_insult_model
#     global logistic_obscene_model
#     global logistic_severe_toxic_model
#     global logistic_threat_model
#     global logistic_toxic_model

with open('/models/tfidf_vectorizer_train.pkl', 'rb') as tfidf_file:
        tfidf_model = pickle.load(tfidf_file)
with open('/models/logistic_identity_hate.pkl', 'rb') as logistic_identity_hate_file:
    logistic_identity_hate_model = pickle.load(logistic_identity_hate_file)
with open('/models/logistic_insult.pkl', 'rb') as logistic_insult_file:
    logistic_insult_model = pickle.load(logistic_insult_file)
with open('/models/logistic_obscene.pkl', 'rb') as logistic_obscene_file:
    logistic_obscene_model = pickle.load(logistic_obscene_file)
with open('/models/logistic_severe_toxic.pkl', 'rb') as logistic_severe_toxic_file:
    logistic_severe_toxic_model = pickle.load(logistic_severe_toxic_file)
with open('/models/logistic_threat.pkl', 'rb') as logistic_threat_file:
    logistic_threat_model = pickle.load(logistic_threat_file)
with open('/models/logistic_toxic.pkl', 'rb') as logistic_toxic_file:
    logistic_toxic_model = pickle.load(logistic_toxic_file)


@app.route('/')
def my_form():
    return render_template('main.html')


@app.route('/', methods=['POST'])
def my_form_post():
    """
        Takes the comment submitted by the user, apply TFIDF trained vectorizer to it, predict using trained models
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
