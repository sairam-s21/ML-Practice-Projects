import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from data_processing import clean_text
import pickle

# Defining input and output files
INPUT_FILE = 'cleaned_imdb_dataset.csv'
OUTPUT_FILE = 'sentiment_model.pkl'

def train_model():
    
    print("--- Starting Model Training ---")
    try:
        #Loading the dataset
        df = pd.read_csv(INPUT_FILE)
        
        # Dividing feature and label columns
        X = df['cleaned_review']
        y = df['sentiment']

        # Creating Training and Testing Sets
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

        # Initializing vectorizer
        vectorizer = CountVectorizer()
        
        # vectorizing the text
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # Raining the model
        model = MultinomialNB()
        model.fit(X_train_vec,y_train)

        # Predicting output for given test set
        predictions = model.predict(X_test_vec)
        
        # Evaluating the model's accuracy
        accuracy = accuracy_score(y_test,predictions)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")

        # Save the Artifacts the artifacts as a dictionary
        artifacts = {'vectorizer': vectorizer, 'model': model}
        
        # Using pickle to save the artifacts into the output file
        with open(OUTPUT_FILE,'wb') as f:
             pickle.dump(artifacts,f)

        print(f"--- Model and Vectorizer saved successfully to {OUTPUT_FILE}! ---")
    except FileNotFoundError:
        print(f"Error: The input file '{INPUT_FILE}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def run_prediction_example():

    #Loads the model and makes prediction on sample reviews
    print("\n--- Running Prediction Example ---")
    try:
        with open(OUTPUT_FILE, 'rb') as f:
            artifacts = pickle.load(f)
        
        vectorizer = artifacts['vectorizer']
        model = artifacts['model']
        
        sample_reviews = [
            "This movie was absolutely fantastic, one of the best I have seen all year!",
            "A complete waste of time. The plot was boring and the acting was terrible.",
            "It was an okay film, not great but not bad either."
        ]
        
        for review in sample_reviews:
            # New, unseen data must go through the same steps: clean and transform
            cleaned_review = clean_text(review)
            review_vec = vectorizer.transform([cleaned_review])
            prediction = model.predict(review_vec)[0]
            
            print(f"\nReview: '{review}'")
            print(f"Predicted Sentiment: {prediction}")
            
    except FileNotFoundError:
        print(f"Could not load '{OUTPUT_FILE}'. Please train the model first by running the script.")
    except Exception as e:
        print(f"An error occurred during prediction: {e}")

# Main function
if __name__ == '__main__':
    train_model()
    run_prediction_example()