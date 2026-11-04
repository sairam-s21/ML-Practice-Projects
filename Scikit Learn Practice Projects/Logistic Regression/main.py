import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_data(path, label_column):
    df=pd.read_csv(path)
    X=df.drop(columns=[label_column])
    y=df[label_column]
    return X,y

def train_model(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
    model=LogisticRegression()
    model.fit(X_train,y_train)
    return model, X_test, y_test

def evaluate_model(model,X_test,y_test):
    y_pred=model.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    precision=precision_score(y_test,y_pred)
    recall=recall_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    return accuracy, precision, recall, f1

def display_results(model, accuracy, precision, recall, f1):
    print("Model used: ",type(model).__name__)
    print(f"Accuracy Score: {accuracy}")
    print(f"Precision Score: {precision}")
    print(f"Recall Score: {recall}")
    print(f"F1 Score: {f1}")

if __name__ == "__main__":
    X, y = load_data("students.csv","Exam_Pass")
    model, X_test, y_test = train_model(X,y)
    accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)
    display_results(model, accuracy, precision, recall, f1)