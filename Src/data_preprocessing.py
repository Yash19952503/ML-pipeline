import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import nltk

# Ensure necessary NLTK resources are available
nltk.download('stopwords')

# Logger setup
def setup_logger(name, log_dir='logs', log_file='data_preprocessing.log'):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(os.path.join(log_dir, log_file))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logger('data_preprocessing')

# Text preprocessing
def transform_text(text, tokenizer, stop_words, ps):
    """
    Transforms input text: tokenization, stopword/punctuation removal, and stemming.
    """
    text = text.lower()  # Lowercase
    tokens = tokenizer.tokenize(text)  # Tokenize with RegexpTokenizer
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return " ".join(ps.stem(word) for word in tokens)  # Stem and join tokens

# Data preprocessing
def preprocess_df(df, text_column='text', target_column='target'):
    """
    Preprocesses the DataFrame: encodes target column, removes duplicates, and transforms the text column.
    """
    try:
        logger.debug('Starting preprocessing for DataFrame')

        # Encode the target column
        if target_column in df.columns:
            df[target_column] = LabelEncoder().fit_transform(df[target_column])
            logger.debug('Target column encoded')

        # Remove duplicates
        df = df.drop_duplicates()
        logger.debug('Duplicates removed')

        # Prepare tokenizer, stopwords, and stemmer
        tokenizer = RegexpTokenizer(r'\w+')  # Tokenizer to extract words only
        stop_words = set(stopwords.words('english'))
        ps = PorterStemmer()

        # Transform text column
        if text_column in df.columns:
            df[text_column] = df[text_column].astype(str).apply(transform_text, tokenizer=tokenizer, stop_words=stop_words, ps=ps)
            logger.debug('Text column transformed')
        else:
            logger.warning(f"{text_column} column not found in DataFrame")

        return df

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise

# Main function
def main(input_dir='./data/raw', output_dir='./data/interim', text_column='text', target_column='target'):
    """
    Main function to load, preprocess, and save data.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        # Load datasets
        train_path, test_path = os.path.join(input_dir, 'train.csv'), os.path.join(input_dir, 'test.csv')
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        logger.debug('Data loaded successfully')

        # Preprocess datasets
        train_processed = preprocess_df(train_data, text_column, target_column)
        test_processed = preprocess_df(test_data, text_column, target_column)

        # Save processed datasets
        train_processed.to_csv(os.path.join(output_dir, 'train_processed.csv'), index=False)
        test_processed.to_csv(os.path.join(output_dir, 'test_processed.csv'), index=False)
        logger.debug(f'Processed data saved to {output_dir}')

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

if __name__ == '__main__':
    main()
