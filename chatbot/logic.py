import os
import pandas as pd
import json
import joblib
import tkinter as tk
from tkinter import filedialog
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from deap import base, creator, tools, algorithms
import numpy as np
import nltk
from nltk.corpus import stopwords
import string

# Ensure NLTK stop words are downloaded
nltk.download('stopwords')

MODEL_PATH = 'trained_model.joblib'

class KnowledgeBase:
    def __init__(self):
        self.model = None
        self.vectorizer = TfidfVectorizer()
        self.conversation_history = []
        self.load_model()
        self.prompt_for_training_files()

    # --- Enhanced Data Preprocessing with NLP ---
    def clean_text(self, text):
        """Clean text data by removing punctuation, converting to lowercase, and removing stop words."""
        stop_words = set(stopwords.words('english'))
        text = text.lower()
        text = ''.join([char for char in text if char not in string.punctuation])
        text = ' '.join([word for word in text.split() if word not in stop_words])
        return text

    def prompt_for_training_files(self):
        """Prompt the user to upload training files in various formats."""
        print("Please upload files to train the knowledge base.")
        root = tk.Tk()
        root.withdraw()  # Hide the main tkinter window
        file_paths = filedialog.askopenfilenames(
            title="Select training files",
            filetypes=[("All Files", "*.*"), ("Text Files", "*.txt"), ("CSV Files", "*.csv"), ("JSON Files", "*.json")]
        )
        
        if not file_paths:
            print("No files selected. Knowledge base cannot be trained.")
            return
        
        for file_path in file_paths:
            _, file_extension = os.path.splitext(file_path)
            file_extension = file_extension.lower()
            with open(file_path, 'r', encoding='utf-8') as file_data:
                if file_extension == '.txt':
                    self.add_data(file_data, 'txt')
                elif file_extension == '.csv':
                    self.add_data(file_path, 'csv')
                elif file_extension == '.json':
                    self.add_data(file_path, 'json')
                else:
                    print(f"Unsupported file type: {file_extension}")

    def add_data(self, file_data, file_type):
        """Process and add data to the knowledge base from various file formats."""
        if file_type == 'txt':
            data = [self.clean_text(line.strip()) for line in file_data if line.strip()]
            self.train_model(data)
        elif file_type == 'csv':
            df = pd.read_csv(file_data)
            questions = df['question'].apply(self.clean_text).tolist()
            answers = df['answer'].apply(self.clean_text).tolist()
            self.train_model(questions, answers)
        elif file_type == 'json':
            with open(file_data, 'r', encoding='utf-8') as f:
                data = json.load(f)
            questions, answers = [], []
            for item in data.get('intents', []):
                for pattern in item['patterns']:
                    questions.append(self.clean_text(pattern))
                    answers.append(self.clean_text(item['responses'][0]))
            self.train_model(questions, answers)

    # --- Cross-Validation for Model Selection ---
    def select_best_model(self, X_train, y_train):
        """Cross-validate multiple models and select the best one based on average score."""
        models = {
            'Naive Bayes': MultinomialNB(),
            'Perceptron': Perceptron(max_iter=1000, tol=1e-3),
            'SVM': SVC(probability=True)
        }
        best_model = None
        best_score = 0
        for name, model in models.items():
            print(f"Evaluating {name}...")
            model.fit(X_train, y_train)
            score = cross_val_score(model, X_train, y_train, cv=5).mean()
            print(f"{name} model cross-validation score: {score}")
            if score > best_score:
                best_score = score
                best_model = model
        print(f"Best model selected: {best_model}")
        return best_model

    # --- Training with Best Model Selection ---
    def train_model(self, questions, answers=None):
        """Train the model using questions and optional answers."""
        if answers is None:
            answers = questions
        X_train, X_test, y_train, y_test = train_test_split(questions, answers, test_size=0.2, random_state=42)

        # Select the best model based on cross-validation
        print("Selecting the best model for training...")
        self.model = self.select_best_model(X_train, y_train)
        
        # Final training with selected model on all data
        print("Training with selected model...")
        self.model.fit(X_train, y_train)

        # Evaluate the model and save it
        self.evaluate_model(X_test, y_test)
        self.save_model()
        print("Training completed and model saved.")

    # --- Model Evaluation ---
    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model and print a performance report."""
        y_pred = self.model.predict(X_test)
        print("Model Evaluation Report:")
        print(classification_report(y_test, y_pred))

    def optimize_hyperparameters(self, questions, answers):
        """Optimize model hyperparameters using genetic algorithms."""

        def evaluate_individual(individual):
            """Evaluate a set of hyperparameters."""
            max_features, ngram_range, alpha = individual
            vectorizer = TfidfVectorizer(max_features=int(max_features), ngram_range=(1, int(ngram_range)))
            model = make_pipeline(vectorizer, MultinomialNB(alpha=alpha))
            X_train, X_test, y_train, y_test = train_test_split(questions, answers, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            score = cross_val_score(model, X_test, y_test, cv=3).mean()
            return score,

        # Genetic Algorithm setup
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        toolbox.register("attr_max_features", np.random.randint, 500, 5000)
        toolbox.register("attr_ngram_range", np.random.randint, 1, 3)
        toolbox.register("attr_alpha", np.random.uniform, 0.01, 1.0)
        
        toolbox.register("individual", tools.initCycle, creator.Individual, 
                         (toolbox.attr_max_features, toolbox.attr_ngram_range, toolbox.attr_alpha), n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        toolbox.register("evaluate", evaluate_individual)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutPolynomialBounded, low=(500, 1, 0.01), up=(5000, 3, 1.0), eta=0.5, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        population = toolbox.population(n=10)
        algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=5, verbose=True)

        best_individual = tools.selBest(population, k=1)[0]
        best_max_features, best_ngram_range, best_alpha = best_individual
        print(f"Best parameters: max_features={best_max_features}, ngram_range=(1, {best_ngram_range}), alpha={best_alpha}")
        
        # Train model with the best parameters
        self.model = make_pipeline(
            TfidfVectorizer(max_features=int(best_max_features), ngram_range=(1, int(best_ngram_range))),
            MultinomialNB(alpha=best_alpha)
        )
        self.model.fit(questions, answers)

    def query(self, query, context=None):
        """Return the predicted response based on the trained model."""
        if not self.model:
            return "I'm not trained yet. Please provide data to train me."
        if context:
            query = context + " " + query
        prediction = self.model.predict([query])
        return prediction[0]

    def save_model(self):
        """Save the trained model to disk."""
        joblib.dump(self.model, MODEL_PATH)
        print("Model saved successfully.")

    def load_model(self):
        """Load the trained model from disk, if it exists."""
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODE_PATH)
            print("Model loaded successfully.")
        else:
            print("No pre-trained model found.")

    def update_context(self, user_input):
        """Store user input in the conversation history for context-aware responses."""
        self.conversation_history.append(user_input)
