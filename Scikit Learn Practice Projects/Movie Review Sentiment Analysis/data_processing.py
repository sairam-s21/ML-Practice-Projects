import pandas as pd
import re
from nltk.corpus import stopwords
import nltk

#storing nltk english stopwords as a set
try:
    STOP_WORDS = set(stopwords.words('english'))
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')
    STOP_WORDS = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    
    #Removing html tags in dataset
    text=re.sub(r'<.*?>','',text)

    #Converting all the words to lowercase
    text=text.lower()

    #Removing all the punctuations
    text=re.sub(r'[^a-z\s]','',text)

    #Splitting all the words and storing as list
    words=text.split()

    #Removing all stopwords such as the,an,is,at, etc ...
    cleaned_words=[word for word in words if word not in STOP_WORDS]

    #Returning all the words as a string
    return ' '.join(cleaned_words)

# Main Function
def main():

    #Defining the input and output files
    INPUT_FILE = 'IMDB Dataset.csv'
    OUTPUT_FILE = 'cleaned_imdb_dataset.csv'

    print(f"Loading data from '{INPUT_FILE}'...")
    try:
        #Loading the dataset
        df = pd.read_csv('IMDB Dataset.csv') 
    except FileNotFoundError:
        print(f"Error: Dataset not found at '{INPUT_FILE}'.")
        print("Please download it and place it in the correct directory.")
        return 

    # Applying the cleaning function to the 'review' column.
    print("Applying text cleaning function to all reviews...")
    df['cleaned_review'] = df['review'].apply(clean_text)

    # Preparing the final DataFrame for saving.
    final_df = df[['cleaned_review','sentiment']]

    # Saving the final DataFrame to OUTPUT_FILE
    print(f"Saving cleaned data to '{OUTPUT_FILE}'...")
    final_df.to_csv(OUTPUT_FILE,index=False)

    print("Data cleaning process complete.")
    print(f"Cleaned data saved to {OUTPUT_FILE}")

# Main function Execution
if __name__ == "__main__":
    main()