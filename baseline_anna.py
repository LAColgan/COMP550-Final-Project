
from __future__ import division
import pandas as pd
from utils import preprocessing as proc
from utils import evaluation as eval
from utils import util as util

CONTRADICTION_THRESHOLD = 0.3
ENTAILMENT_TRESHOLD = 0.6
class Anna_baseline:
    def __init__(self) -> None:
        # read the csv file into data frame, use sentence1, sentence2 and gold_label
        self.dev_df=pd.read_csv("data/snli_1.0_dev.txt", sep='\t')
        #self.test_df=pd.read_csv("data/snli_1.0_test.txt", sep='\t')
        #self.train_df=pd.read_csv("data/snli_1.0_train.txt", sep='\t')

        # Preprocessing of the development set
        self.preprocess(self.dev_df)#, self.preprocess(self.test_df), self.preprocess(self.train_df)
        
        # UNUSED
        sentence_set_pair_list = list(zip(self.dev_df['sentence1'], self.dev_df['sentence2']))


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
        if overlap >= ENTAILMENT_TRESHOLD:
            entailment_response = "entailment"

        return entailment_response
    




if __name__=='__main__':
    Baseline_model=Anna_baseline()

    df=util.add_column(Baseline_model.dev_df, 
                       'classification', 
                       [Baseline_model.compute_classification(s1, s2) 
                        for s1, s2 in zip(Baseline_model.dev_df['sentence1'], Baseline_model.dev_df['sentence2'])])


    # Convert 'classification' and 'gold_label' columns to strings if they're not already
    #df['classification'] = df['classification'].astype(str)
    #df['gold_label'] = df['gold_label'].astype(str)

    # Compute accuracy
    accuracy = eval.get_accuracy(df['classification'], df['gold_label'])

    # Display the accuracy
    print("Accuracy:", accuracy)