# reading data
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV


data = pd.read_csv('data.csv')
# drop unnecessary columns and rename cols
data.drop(['index', 'title'], axis=1, inplace=True)
# data.head()

# # check missing values
# data.isna().sum()
# # check data shape
# data.shape
# # check target balance
# plt.plot(data['genre'].value_counts(normalize=True))

# convert text to lowercase
data['summary'] = data['summary'].str.lower()
# split the dataset into features (X) and labels (y)
X = data['summary']
y = data['genre']


def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]
    lemmatize = WordNetLemmatizer()
    lemmatized_tokens = [lemmatize.lemmatize(token) for token in filtered_tokens]
    return " ".join(lemmatized_tokens)


# pre-process the text input
X_preprocessed = [preprocess_text(text) for text in X]

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_preprocessed, y, test_size=0.2, random_state=42)

# transform text in numerical values using TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
# tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


# train the classification model

# model = MultinomialNB()
# model.fit(X_train_tfidf, y_train)
# accuracy: 0.42 ~ 0.45

# clf = RidgeClassifier(tol=1e-2, solver="sparse_cg")
# clf.fit(X_train_tfidf, y_train)
# accuracy: 0.70

# 0.98 train score
# 0.58 accuracy with a standard deviation of 0.03
clf = RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1])
clf.fit(X_train_tfidf, y_train)
train_score = clf.score(X_train_tfidf, y_train)
print("%0.2f train score" % train_score)

# make predictions
# y_pred = model.predict(X_test_tfidf)
pred = clf.predict(X_test_tfidf)

# evaluate the model
# Calculate accuracy and generate a classification report
scores = cross_val_score(clf, X_test_tfidf, y_test, cv=5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
# test_score = clf.score(X_test_tfidf, y_test)
# print("%0.2f test score" % test_score)
# accuracy = clf.score(X_test_tfidf, y_test)
# print(f"Accuracy: {accuracy:.2f}")

report = classification_report(y_test, pred)
print(report)


# nltk.download("punkt")
# nltk.download("stopwords")
# nltk.download("wordnet")