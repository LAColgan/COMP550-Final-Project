import models
from utils import util
from utils import evaluation as eval
from models.BERT import LoadData, FineTune, Test
from models import overlapBaseline
import os
from models.tests import grade_bert


if __name__ == '__main__':

    # Baseline model
    print('baseline model')
    Baseline_model = overlapBaseline.OverlapBaseline('../data/snli_1.0_test.txt')
    df = util.add_column(Baseline_model.dev_df,
                            'classification',
                            [Baseline_model.compute_classification(s1, s2)
                            for s1, s2 in zip(Baseline_model.dev_df['sentence1'],
                            Baseline_model.dev_df['sentence2'])])
    # Convert 'classification' and 'gold_label' columns to strings if they're not already
    df['classification'] = df['classification'].astype(str)
    df['gold_label'] = df['gold_label'].astype(str)
    # Compute accuracy
    accuracy = eval.get_accuracy(df['classification'], df['gold_label'])
    # Display the accuracy
    print("Accuracy:", accuracy)
    #test on summaries data
    print("Accuracy on test set (students summary): ")
    print(models.tests.test_baseline())

    # BERT model (finetuned)
    print("Bert model")
    train_data_file = os.path.join("data", "snli_1.0_train.txt")
    dev_data_file = os.path.join("data", "snli_1.0_dev.txt")
    val_data_file = os.path.join("data", "snli_1.0_test.txt")
    # Load data
    train_loader = LoadData(train_data_file).load()
    dev_loader = LoadData(dev_data_file).load()
    val_loader = LoadData(val_data_file).load()
    # Finetune
    model = FineTune(train_loader, dev_loader).finetune()
    test_acc, test_loss = Test(val_loader).test()

    # Test models on simulated student reports
    bert_grading = grade_bert()
