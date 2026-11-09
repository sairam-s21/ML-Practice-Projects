import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def select_features(df: pd.DataFrame) -> pd.DataFrame:
   
    # Selecting relevant features
    features = ['Age at enrollment','Gender','Debtor','Tuition fees up to date','Curricular units 1st sem (approved)']
    X = df[features]
    return X


def handle_missing_values(X: pd.DataFrame) -> pd.DataFrame:

    X = X.copy()

    # Filling missing values in categorical columns with 'Unknown'
    categorical_cols = ['Gender','Debtor','Tuition fees up to date']
    X.loc[:,categorical_cols]=X[categorical_cols].fillna('Unknown')
    # Filling missing values in numerical columns with the median
    numerical_cols = ['Age at enrollment','Curricular units 1st sem (approved)']
    X.loc[:,numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].median())
    return X


def encode_categorical_features(X: pd.DataFrame) -> pd.DataFrame:
    
    # One-hot encoding categorical features
    categorical_cols = ['Gender','Debtor','Tuition fees up to date']
    X=pd.get_dummies(X,columns=categorical_cols,drop_first=True)
    return X


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.3) -> tuple:
    
    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,random_state=42)
    return X_train, X_test, y_train, y_test


def train_decision_tree(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
    
    # Training DecisionTreeClassifier model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train,y_train)
    return model


def evaluate_model(model: DecisionTreeClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> tuple:
    
    # Evaluating model predictions and accuracy
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test,y_test)
    cm = confusion_matrix(y_test,y_pred)
    return accuracy, cm


def apply_l2_regularization(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> LogisticRegression:
    
    # Scaling the data and training the L2-regularized model
    scaler=StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)

    X_train_scaled=pd.DataFrame(X_train_scaled,columns=X_train.columns, index=X_train.index)
    X_test_scaled=pd.DataFrame(X_test_scaled,columns=X_test.columns,index=X_test.index)
    reg_model_l2=LogisticRegression(penalty='l2',random_state=42,max_iter=2000)
    reg_model_l2.fit(X_train_scaled,y_train)

    return reg_model_l2


def apply_l1_regularization(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> LogisticRegression:
   
    # Scaling the data and training the L1-regularized model
    scaler=StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    X_train_scaled=pd.DataFrame(X_train_scaled,columns=X_train.columns,index=X_train.index)
    X_test_scaled=pd.DataFrame(X_test_scaled,columns=X_test.columns,index=X_test.index)
    reg_model_l1=LogisticRegression(penalty='l1',solver='saga',random_state=42,max_iter=2000)
    reg_model_l1.fit(X_train_scaled,y_train)
    return reg_model_l1


# Main Execution Block
if __name__ == '__main__':
    file_path = 'dataset.csv'
    
    df = pd.read_csv(file_path)
    
    X = select_features(df)
    X = handle_missing_values(X)
    X = encode_categorical_features(X)
    y = df['Target']
    
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    model = train_decision_tree(X_train, y_train)
    accuracy, cm = evaluate_model(model, X_test, y_test)
    print("\nDecision Tree Model Accuracy: ", accuracy)
    print("\nConfusion Matrix:\n", cm)
    
    reg_model_l2 = apply_l2_regularization(X_train, y_train, X_test, y_test)
    accuracy_l2 = reg_model_l2.score(X_test, y_test)
    print("\nL2 Regularized Model Accuracy (Ridge): ", accuracy_l2)
    
    reg_model_l1 = apply_l1_regularization(X_train, y_train, X_test, y_test)
    accuracy_l1 = reg_model_l1.score(X_test, y_test)
    print("\nL1 Regularized Model Accuracy (Lasso): ", accuracy_l1)
