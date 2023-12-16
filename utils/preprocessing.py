from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter


def remove_stop_words(df, threshold):
    df['combined'] = df['sentence1'] + " " + df['sentence2']

    # Tokenize and get stop words from combined text
    def get_stop_words(sentences):
        words = [word_tokenize(sentence.lower()) for sentence in sentences]
        words = [word for sublist in words for word in sublist]
        stop_words = set(stopwords.words('english'))
        word_counts = Counter(words)
        stop_words_to_remove = [word for word, count in word_counts.items() if
                                word in stop_words and count > threshold]
        return set(stop_words_to_remove)

    # Get stop words from the combined column
    stop_words_combined = get_stop_words(df['combined'])

    # Function to remove stop words from sentence1 and sentence2
    def remove_stop_words_from_sentences(text):
        words = word_tokenize(text.lower())
        return ' '.join(word for word in words if word not in stop_words_combined)

    # Apply stop word removal to sentence1 and sentence2
    df['sentence1'] = df['sentence1'].apply(remove_stop_words_from_sentences)
    df['sentence2'] = df['sentence2'].apply(remove_stop_words_from_sentences)
    return df

def remove_stop_words_for_test(df, title, threshold):

    # Tokenize and get stop words from combined text
    def get_stop_words(sentences):
        words = [word_tokenize(sentence.lower()) for sentence in sentences]
        words = [word for sublist in words for word in sublist]
        stop_words = set(stopwords.words('english'))
        word_counts = Counter(words)
        stop_words_to_remove = [word for word, count in word_counts.items() if
                                word in stop_words and count > threshold]
        return set(stop_words_to_remove)

    # Get stop words from the combined column
    stop_words = get_stop_words(df[title])

    def remove_stop_words_from_sentences(text):
        words = word_tokenize(text.lower())
        return ' '.join(word for word in words if word not in stop_words)

    # Apply stop word removal to sentence1 and sentence2
    df[title] = df[title].apply(remove_stop_words_from_sentences)
    return df