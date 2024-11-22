from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from KFoldXValidation import KFoldXValidationClassification

from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

#get text file
link_data = "***\\Data\\SMSSpamCollection"
df = pd.read_csv(link_data, sep="\t",header=None , names=["label", "message"])

#Hyperparameter
test_size = 0.2

#split the date into train an d test data (it is important to split the text by random)
X_train, X_test, y_train, y_test = train_test_split(df["message"], df["label"], test_size=test_size)

def format_data(X_train, X_test):
    """
    
    The function formats the provided train and test data by converting each message to lowercase,
    tokenizing it into a list, and applying a Snowball stemmer to remove morphological affixes, leaving only the word stem.
    Additionally, it removes all stopwords from the messages, and then creates a vocabulary and vector representation for the training messages.

    Args:
        X_train (Pandas Series): The splited Data from the original Data to train the model.
        X_test (Pandas Series): The splited Data from the original Data to test the model.

    Returns:
        X_train (Pandas DataFrame): The formatted Data to train the Model.
        X_test (Pandas DataFrame): The formatted Data to test the Model.
        vocabulary (list): Vocabulary of the train Dataset.
        vector (list[int[0]]): Vecotor with the length of the vocabulary. Initialised with 0's.
    """
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)

    #lower the message of every row and format it into a list
    #also use a snowballsteller that remove morphological affixes from words, leaving only the word stem.
    #also delete all the stopwords in the message
    stemmer = SnowballStemmer("english")
    X_train["words"] = X_train["message"].apply(lambda x: word_tokenize(x.lower()))
    X_train["words"] = X_train["words"].apply(lambda words: [stemmer.stem(word) for word in words if word.lower() not in ENGLISH_STOP_WORDS])
    X_test["words"] = X_test["message"].apply(lambda x: word_tokenize(x.lower()))
    X_test["words"] = X_test["words"].apply(lambda words: [stemmer.stem(word) for word in words if word.lower() not in ENGLISH_STOP_WORDS])

    #creating vocabulary and vector for the train messages
    vocabulary = list(set( X_train["words"].sum()))
    vector = [0 for _ in vocabulary]

    return X_train, X_test, vocabulary, vector

#implementing bow
def create_vecotor(message):
    """
    Creates the vector for the bow. Index of vektor is the index of every unique word in vocabulary.
    This Vektor index gets a 1 if the message has the word in it.

    Args:
        message (list[str]): The message to create the vector from.

    Returns:
        (list[int]): The vektor of bow.
    """
    vector_copy = vector[:]
    for word in message:
        #check if word is in vocabulary
        #is important for the test_data because not every word in the test_data is in the vocabulary of the train data
        if word in vocabulary:
            vector_copy[vocabulary.index(word)]=1
    return vector_copy

def bag_of_words(documents):
    """
    Apllys on every row of panda Series data create_vecotor(documents). So Every row now has its bow for further use.

    Args:
        documents (Pandas Series): All the Data to create the bow vector for.

    Returns:
        (Pandas Series): A Series that has all the vectors for the rows.
    """
    return documents.apply(lambda documents: create_vecotor(documents))

#format Train data
X_train, X_test, vocabulary, vector = format_data(X_train, X_test)
#creating a list of BoW(vector) for every row in the Data
x_train_bow  = list(bag_of_words(X_train["words"]))
#creating a list of tf-idf(vector) for every row in the Data
Y_train_list = list(y_train)

ten_fold_model = KFoldXValidationClassification()
ten_fold_model.fit(x_train_bow,Y_train_list)

X_test_words = X_test["words"]
accuracy = accuracy_score(y_test, ten_fold_model.predict(list(bag_of_words(X_test_words))))

print(f"Error test bow: {1-ten_fold_model.getAccuracy()}")