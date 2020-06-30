from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

# DIFFERENT VECTORIZERS
 tfidf_Vect = TfidfVectorizer()
# tfidf_Vect = TfidfVectorizer(ngram_range=(1,2))
# tfidf_Vect = TfidfVectorizer(stop_words='english')

X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)  # GET DATA TO TEST

# print(tfidf_Vect.vocabulary_)
clf = MultinomialNB()
clf.fit(X_train_tfidf, twenty_train.target)  # TRAIN DATA

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)  # CREATE TEST GROUP
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)  # TRANSFORM DATA

predicted = clf.predict(X_test_tfidf)  # GET PREDICTION

score = metrics.accuracy_score(twenty_test.target, predicted)  # TEST ACCURACY
print('This test is for standard vectorizer\n')
print("Accuracy for NB-> ", score)

# THIS SECTION IS FOR SVC
clf2 = SVC(kernel='linear')
clf2.fit(X_train_tfidf, twenty_train.target)  # TRAIN DATA
y_predict = clf2.predict(X_test_tfidf)  # GET PREDICTION
score2 = metrics.accuracy_score(twenty_test.target, y_predict)  # TEST ACCURACY
print("Accuracy for SVM -> ", score2)
