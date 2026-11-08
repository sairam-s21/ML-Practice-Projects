import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# from sklearn.ensemble import RandomForestClassifier  # Optional upgrade

def preprocess(df):
    # Filling missing numerical values with the median
    df = df.fillna(df.median(numeric_only=True))
    return df

def train_and_evaluate():
    # Loading the training dataset
    train_data = pd.read_csv("Titanic_train.csv")
    
    # Separating features and target
    y_train = train_data["Survived"]
    X_train = preprocess(train_data.drop(columns=["Survived"]))
    
    # Training DecisionTreeClassifier model
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Loading the test dataset
    test_data = pd.read_csv("Titanic_test.csv")
    X_test = preprocess(test_data)
    
    # Predicting survival outcomes
    preds = model.predict(X_test)
    
    return preds

if __name__ == "__main__":
    preds = train_and_evaluate()
    
    # Calculating model accuracy
    true_labels = pd.read_csv("Titanic_test_labels.csv")["Survived"]
    acc = accuracy_score(true_labels, preds)
    
    print(f"The model accuracy: {acc*100:.2f}%")
