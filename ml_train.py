import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression

from utils import tokenize


def fit_logistic(x, y):
    y = y.values
    model = LogisticRegression(C=4, dual=True)
    return model.fit(x, y)


if __name__ == '__main__':

    # data is from here https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
    PATH = "/Users/palermospenano/Desktop/Dropbox/data_science/kaggle/toxic_comment_kaggle_v2/data"

    COMMENT = 'comment_text'
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    print("Load data...")
    train = pd.read_csv('{}/train.csv'.format(PATH))
    test = pd.read_csv('{}/test.csv'.format(PATH))

    print("Fill empty with unknown...")
    train[COMMENT].fillna('unknown', inplace=True)
    test[COMMENT].fillna('unknown', inplace=True)

    print("Train TFIDF vectorizer...")
    tfidfvectorizer = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenize,
                                      min_df=3, max_df=0.9, strip_accents='unicode',
                                      use_idf=1, smooth_idf=True, sublinear_tf=1)

    train_term_doc = tfidfvectorizer.fit_transform(train[COMMENT])
    x = train_term_doc

    joblib.dump(tfidfvectorizer, 'models/tfidf_vectorizer_train.pkl')

    print("Fit logistic regression for each class...")
    for i, j in enumerate(label_cols):
        print("Fitting:", j)
        model = fit_logistic(x, train[j])
        joblib.dump(model, 'models/logistic_{}.pkl'.format(j))
