import pandas as pd

class Sequential:
    def __init__(self) -> None:
        self.dev_df=pd.read_csv("data/snli_1.0_dev.txt", sep='\t')
        self.test_df=pd.read_csv("data/snli_1.0_test.txt", sep='\t')
        self.train_df=pd.read_csv("data/snli_1.0_train.txt", sep='\t')