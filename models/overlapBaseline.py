from __future__ import division
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from utils import preprocessing as proc

CONTRADICTION_THRESHOLD = 0.2
ENTAILMENT_THRESHOLD = 0.7


class OverlapBaseline:
    def __init__(self, file_path) -> None:
        self.dev_df=pd.read_csv(file_path, sep='\t')
        # Preprocessing data set
        self.preprocess(self.dev_df)

    def preprocess(self, df, remove_stopwords=True):
        if remove_stopwords:
            df=proc.remove_stop_words(df, 3)
    
    def compute_basic_classification(self, s1, s2):
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

    def compute_classification_BLEU(self, s1, s2):
        reference = [s1.lower().split()]
        candidate = s2.lower().split()

        smoothing = SmoothingFunction().method4
        bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothing)

        entailment_response = "contradiction"
        if bleu_score > CONTRADICTION_THRESHOLD:
            entailment_response = "neutral"
        if bleu_score >= ENTAILMENT_THRESHOLD:
            entailment_response = "entailment"
        return entailment_response