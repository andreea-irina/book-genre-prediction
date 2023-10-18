# reading data
import matplotlib
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('data.csv')
# drop unnecessary columns and rename cols
data.drop(['index', 'title'], axis=1, inplace=True)
# data.head()

# # check missing values
data.isna().sum()
# # check data shape
# data.shape
# # check target balance
# data['genre'].value_counts(normalize=True).plot.bar()

# convert text to lowercase
data['summary'] = data['summary'].str.lower()
# split the dataset into features (X) and labels (y)
X = data['summary']
y = data['genre']


def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    return " ".join(stemmed_tokens)


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
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# make predictions
y_pred = model.predict(X_test_tfidf)

# evaluate the model
# Calculate accuracy and generate a classification report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

report = classification_report(y_test, y_pred)
print(report)


# nltk.download("punkt")
# nltk.download("stopwords")