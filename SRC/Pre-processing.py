import os
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk

nltk.download('stopwords')
nltk.download('punkt')


# Ensure the log directory exists
log_dirs = 'logs'
os.makedirs(log_dirs, exist_ok=True)


# Set up logger
logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dirs, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def transform_text(text):
    """Transform text by lowering, tokenizing, removing stopwords
    and punctuation, and stemming."""

    ps = PorterStemmer()

    # Lower case
    text = text.lower()

    # Tokenize
    text = nltk.word_tokenize(text)

    # Remove non-alphabetical tokens
    text = [word for word in text if word.isalnum()]

    # Remove stopwords
    text = [
        word for word in text
        if word not in stopwords.words('english')
    ]

    # Stem
    text = [ps.stem(word) for word in text]

    # Join tokens
    return ' '.join(text)


def preprocess_df(df, text_column='text', target_column='target'):
    """Preprocess dataframe."""

    try:
        logger.debug("Starting preprocessing of dataframe")

        # Encode target column
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])

        logger.debug("Target column encoded")

        # Remove duplicates
        df = df.drop_duplicates(keep='first')

        logger.debug("Duplicates removed")

        # Transform text
        df.loc[:, text_column] = df[text_column].apply(transform_text)

        logger.debug("Text transformation done")

        return df

    except KeyError as e:
        logger.exception(f"Column not found: {e}")
        raise

    except Exception as e:
        logger.exception("Some unknown error occurred")
        raise


def main(text_column='text', target_column='target'):

    try:

        # Fetch data from data/raw
        train_data = pd.read_csv('data/raw/train.csv')
        test_data = pd.read_csv('data/raw/test.csv')

        logger.debug("Data loaded successfully")

        # Transform data
        train_processed_data = preprocess_df(
            train_data,
            text_column,
            target_column
        )

        test_processed_data = preprocess_df(
            test_data,
            text_column,
            target_column
        )

        # Store processed data
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(
            os.path.join(data_path, "train_processed_data.csv"),
            index=False
        )

        test_processed_data.to_csv(
            os.path.join(data_path, "test_processed_data.csv"),
            index=False
        )

        logger.debug(f"Processed data saved to {data_path}")

    except FileNotFoundError as e:
        logger.exception(f"File not found: {e}")
        raise

    except pd.errors.ParserError as e:
        logger.exception(f"Unable to parse the data: {e}")
        raise

    except Exception as e:
        logger.exception("Unable to complete data preprocessing process")
        raise


if __name__ == '__main__':
    main()