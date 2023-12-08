
from __future__ import division
import pandas as pd

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


# read the csv file into data frame, use sentence1, sentence2 and gold_label
df = pd.read_csv("snli_1.0_dev.txt", sep='\t')
df = pd.DataFrame(df)

remove_stop_words(df, 3)
sentence_set_pair_list = list(zip(df['sentence1'], df['sentence2']))

contradiction_threshold = 0.3
entailment_threshold = 0.6

# Function to compute classification
def compute_classification(s1, s2):
    words_s1 = set(s1.lower().split())
    words_s2 = set(s2.lower().split())

    if max(len(words_s1), len(words_s2)) == 0:
        overlap = 1
    else:
        overlap = len(words_s1 & words_s2) / max(len(words_s1), len(words_s2))

    entailment_response = "contradiction"
    if overlap > contradiction_threshold:
        entailment_response = "neutral"
    if overlap >= entailment_threshold:
        entailment_response = "entailment"

    return entailment_response

# Create a new column 'classification' using the compute_classification function
df['classification'] = [compute_classification(s1, s2) for s1, s2 in zip(df['sentence1'], df['sentence2'])]

# Function to compute accuracy
def compute_accuracy(predictions, true_labels):
    correct_predictions = sum(predictions == true_labels)
    total_predictions = len(predictions)
    accuracy = correct_predictions / total_predictions
    return accuracy

# Convert 'classification' and 'gold_label' columns to strings if they're not already
#df['classification'] = df['classification'].astype(str)
#df['gold_label'] = df['gold_label'].astype(str)

# Compute accuracy
accuracy = compute_accuracy(df['classification'], df['gold_label'])

# Display the accuracy
print("Accuracy:", accuracy)