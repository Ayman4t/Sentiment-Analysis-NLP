from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from nltk.stem import PorterStemmer
import joblib
from sklearn.linear_model import LogisticRegression
import re
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb



# Set the path to the directory containing the positive and negative reviews
dir_path = "txt_sentoken"

def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)

    # Define custom list of stopwords with negations
    custom_stop_words = {'but', 'isn', 'don', 'hasn', 'haven', 'shouldn', 'wasn', 'weren', 'won', 'wouldn', 'no', 'not', 'nor', 'ain'}

    # Remove punctuation, non-alphabetic characters, and exclude custom stopwords
    tokens = [re.sub(r'[^\w\s]', '', word.lower()) for word in tokens if word.isalpha() and word.lower() not in custom_stop_words]

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Stemming
    # stemmer = PorterStemmer()
    # tokens = [stemmer.stem(word) for word in tokens]

    return " ".join(tokens)

# Read the positive and negative reviews
pos_files = os.listdir(os.path.join(dir_path, 'pos'))
neg_files = os.listdir(os.path.join(dir_path, 'neg'))

pos_reviews = []
neg_reviews = []

for file in pos_files:
    with open(os.path.join(dir_path, 'pos', file), 'r') as f:
        pos_reviews.append(preprocess_text(f.read()))

for file in neg_files:
    with open(os.path.join(dir_path, 'neg', file), 'r') as f:
        neg_reviews.append(preprocess_text(f.read()))

# Create labels
pos_labels = [1] * len(pos_reviews)
neg_labels = [0] * len(neg_reviews)

# Combine positive and negative reviews and labels
all_reviews = pos_reviews + neg_reviews
all_labels = pos_labels + neg_labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(all_reviews, all_labels, test_size=0.15, random_state=42,shuffle=True)

# Convert text data to TF-IDF features with additional parameters
vectorizer = TfidfVectorizer(max_features=5000, sublinear_tf=True, ngram_range=(1, 4))

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Logistic Regression


# Define and train the Logistic Regression classifier
logistic_regression = LogisticRegression(max_iter=12, C=2.5)
logistic_regression.fit(X_train_tfidf, y_train)

y_train_pred = logistic_regression.predict(X_train_tfidf)

# Calculate training accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Training Accuracy:", train_accuracy * 100)

# Save the trained model to a file
model_filename = 'logistic_regression_model.pkl'
joblib.dump(logistic_regression, model_filename)
print(f"Trained model saved as '{model_filename}'")

# Make predictions
y_pred = logistic_regression.predict(X_test_tfidf)

# Evaluate the model
print("test Accuracy:", accuracy_score(y_test, y_pred) * 100)
print("Classification Report:")
print(classification_report(y_test, y_pred))
# test Accuracy: 86 %


# Support Vector Machine


# # Train a Support Vector Machine (SVM) classifier
# svm_classifier =SVC(kernel='linear', C=2.4)
# svm_classifier.fit(X_train_tfidf, y_train)

# model_filename = 'Support_Vector_classifier_model.pkl'
# joblib.dump(svm_classifier, model_filename)
# print(f"Trained Support Vector classifier model saved as '{model_filename}'")
# # Make predictions on the training set
# y_train_pred = svm_classifier.predict(X_train_tfidf)

# # Calculate training accuracy
# train_accuracy = accuracy_score(y_train, y_train_pred)
# print("Training Accuracy:", train_accuracy * 100)

# # Make predictions on the test set
# y_test_pred = svm_classifier.predict(X_test_tfidf)

# # Evaluate the model on the test set
# print("Test Accuracy:", accuracy_score(y_test, y_test_pred) * 100)
# print("Classification Report:")
# print(classification_report(y_test, y_test_pred))
# test Accuracy: 85 %


# RandomForestClassifier


# # Define and train the Random Forest classifier
# rf_classifier = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
# rf_classifier.fit(X_train_tfidf, y_train)

# # # Save the trained model to a file
# model_filename = 'random_forest_classifier_model.pkl'
# joblib.dump(rf_classifier, model_filename)
# print(f"Trained Random Forest classifier model saved as '{model_filename}'")

# # Make predictions
# y_pred = rf_classifier.predict(X_test_tfidf)
# y_pred_train = rf_classifier.predict(X_train_tfidf)

# # Evaluate the model
# print("test Accuracy:", accuracy_score(y_test, y_pred) * 100)
# print("Classification Report:")
# print(classification_report(y_test, y_pred))

# print("train Accuracy:", accuracy_score(y_train, y_pred_train) * 100)
# print("Classification Report:")
# print(classification_report(y_train, y_pred_train))

# test Accuracy: 81 %


# Gradient Boosting Machine


# # Train a Gradient Boosting Machine (GBM) classifier
# gbm_classifier = xgb.XGBClassifier(max_depth=10)
# gbm_classifier.fit(X_train_tfidf, y_train)

# model_filename = 'Gradient_Boosting_classifier_model.pkl'
# joblib.dump(gbm_classifier, model_filename)
# print(f"Trained Gradient Boosting classifier model saved as '{model_filename}'")
# # Make predictions
# y_pred_test = gbm_classifier.predict(X_test_tfidf)
# y_pred_train = gbm_classifier.predict(X_train_tfidf)

# # Evaluate the model
# print("test Accuracy:", accuracy_score(y_test, y_pred_test) * 100)
# print("Classification Report:")
# print(classification_report(y_test, y_pred_test))


# print("train Accuracy:", accuracy_score(y_train, y_pred_train) * 100)
# print("Classification Report:")
# print(classification_report(y_train, y_pred_train))
# test Accuracy: 80 %




