from models import baseline_anna as Anna, sequential
from utils import util
from utils import evaluation as eval




if __name__=='__main__':

    # Anna's Baseline
    Baseline_model=Anna.Anna_baseline()

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


