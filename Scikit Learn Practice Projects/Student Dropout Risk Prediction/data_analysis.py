import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    
    # Loading the dataset
    df = pd.read_csv(file_path)
    return df


def explore_data(df: pd.DataFrame):
    
    # Printing the dataset shape
    print(df.shape)

    # Printing data types of each column
    print(df.dtypes)

    # Printing the number of missing values
    print(df.isnull().sum())
    
    # Printing value counts for categorical columns
    categorical_columns=['Gender','Course','Tuition fees up to date','Debtor']
    for col in categorical_columns:
        print(f"\n{col} Distribution:\n",df[col].value_counts())

def create_approval_rate(df: pd.DataFrame):
    
    # Creating the approval rate feature
    df['approval_rate']=df['Curricular units 1st sem (approved)']/df['Curricular units 1st sem (enrolled)']
    return df

def create_performance_score(df: pd.DataFrame):
   
    # Creating the performance score feature
    df['performance_score']=df['Curricular units 1st sem (approved)']/df['Curricular units 1st sem (evaluations)']
    return df

def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    
    # Creating engineered features
    df = create_approval_rate(df)
    df = create_performance_score(df)
    
    # Printing the first few rows to verify the engineered features
    print("\nData with new columns: ")
    print(df.head())
    
    return df

# Main Execution Block
if __name__ == '__main__':

    file_path = 'dataset.csv'
    df = load_data(file_path)
    explore_data(df)
    df = create_engineered_features(df)
