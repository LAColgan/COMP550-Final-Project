
from __future__ import division
import pandas as pd
from nltk import sent_tokenize

from utils import preprocessing as proc
from utils import evaluation as eval
from utils import util as util

CONTRADICTION_THRESHOLD = 0.3
ENTAILMENT_THRESHOLD = 0.6


class OverlapBaseline:
    def __init__(self, file_path) -> None:
        self.dev_df=pd.read_csv(file_path, sep='\t')
        # Preprocessing data set
        self.preprocess(self.dev_df)

    def preprocess(self, df, remove_stopwords=True):
        if remove_stopwords:
            df=proc.remove_stop_words(df, 3)
    
    def compute_classification(self, s1, s2):
        words_s1 = set(s1.lower().split())
        words_s2 = set(s2.lower().split())

        if max(len(words_s1), len(words_s2)) == 0:
            overlap = 1
        else:
            overlap = len(words_s1 & words_s2) / max(len(words_s1), len(words_s2))

        entailment_response = "contradiction"
        if overlap > CONTRADICTION_THRESHOLD:
            entailment_response = "neutral"
        if overlap >= ENTAILMENT_THRESHOLD:
            entailment_response = "entailment"

        return entailment_response


def compute_classification(s1, s2):
    words_s1 = set(s1.lower().split())
    words_s2 = set(s2.lower().split())

    if max(len(words_s1), len(words_s2)) == 0:
        overlap = 1
    else:
        overlap = len(words_s1 & words_s2) / max(len(words_s1), len(words_s2))

    entailment_response = "contradiction"
    if overlap > CONTRADICTION_THRESHOLD:
        entailment_response = "neutral"
    if overlap >= ENTAILMENT_THRESHOLD:
        entailment_response = "entailment"

    return entailment_response


