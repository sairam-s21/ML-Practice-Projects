import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_credit_card_fraud_model():

    #loading the dataset
    df = pd.read_csv("creditcard.csv")  

    # Diividing features and label columns
    X = df.drop("Class", axis=1)  
    y = df["Class"]          

    #Scaling Features to [0, 1] 
    scaler = MinMaxScaler()                  
    X_scaled = scaler.fit_transform(X)  

    #Train-Test Split 
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y 
    )

    #Defining Models 
    models = {
        "Multinomial Naive Bayes":MultinomialNB(),  
        "Decision Tree":DecisionTreeClassifier(),   
        "Random Forest":RandomForestClassifier()    
    }

    #  Training and Evaluating models
    for name, model in models.items():
        model.fit(X_train, y_train)          
        preds = model.predict(X_test)  
        acc = accuracy_score(y_test, preds)       
        print(f"{name} Accuracy: {acc*100:.2f}%") 



    # Returning all Train and test variables 
    return X_train, X_test, y_train, y_test, X_scaled, models, scaler


#main function
if __name__ == "__main__":
    train_credit_card_fraud_model()