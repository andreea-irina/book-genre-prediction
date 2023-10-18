# reading data
# import matplotlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('data.csv')

# drop unnecessary columns and rename cols
data.drop(['index', 'title'], axis=1, inplace=True)
# data.head()

# # check missing values
# data.isna().sum()

# # check data shape
# data.shape

# # check target balance
# data['genre'].value_counts(normalize=True).plot.bar()

# Step 2: Text Preprocessing
# Convert text to lowercase
data['summary'] = data['summary'].str.lower()

# Step 3: Split the dataset into features (X) and labels (y)
X = data['summary']
y = data['genre']

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Step 5: Feature Extraction using TF-IDF Vectorization
# You can adjust the max_features parameter
tfidf_vectorizer = TfidfVectorizer(max_features=4000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Step 6: Build the Machine Learning Model
# Instantiate the Multinomial Naive Bayes model
model = MultinomialNB()

# Train the model on the training data
model.fit(X_train_tfidf, y_train)

# Step 7: Make Predictions
# Use the trained model to make predictions on the test data
y_pred = model.predict(X_test_tfidf)

# Step 8: Evaluate the Model
# Calculate accuracy and generate a classification report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

report = classification_report(y_test, y_pred)
print(report)
