import os
import pandas as pd
import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

MODEL_PATH = 'trained_model.joblib'

class KnowledgeBase:
    def __init__(self):
        self.model = None
        self.vectorizer = TfidfVectorizer()
        self.conversation_history = []
        self.load_model()  # Load the model if it already exists

    def add_data(self, file_data, file_type):
        """Process the uploaded file and add data to the knowledge base."""
        if file_type == 'txt':
            data = [line.strip() for line in file_data if line.strip()]
            self.train_model(data)
        elif file_type == 'csv':
            df = pd.read_csv(file_data)
            self.train_model(df['question'], df['answer'])
        elif file_type == 'json':
            data = json.load(file_data)
            questions, answers = [], []
            for item in data.get('intents', []):
                for pattern in item['patterns']:
                    questions.append(pattern)
                    answers.append(item['responses'][0])
            self.train_model(questions, answers)

    def train_model(self, questions, answers=None):
        """Train the model on the questions and answers."""
        if answers is None:
            answers = questions

        self.model = make_pipeline(TfidfVectorizer(), MultinomialNB())
        X_train, X_test, y_train, y_test = train_test_split(questions, answers, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        self.save_model()

    def query(self, query, context=None):
        """Return the predicted response from the trained model."""
        if not self.model:
            return "Sorry, I don't have enough data to answer that."

        if context:
            query = context + " " + query

        prediction = self.model.predict([query])
        return prediction[0]

    def save_model(self):
        """Save the trained model to disk."""
        joblib.dump(self.model, MODEL_PATH)

    def load_model(self):
        """Load the trained model from disk, if it exists."""
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
            print("Model loaded successfully.")
        else:
            print("No pre-trained model found.")
    
    def update_context(self, user_input):
        """Store context for future queries."""
        self.conversation_history.append(user_input)
