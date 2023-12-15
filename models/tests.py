import torch
from transformers import BertForSequenceClassification
import logging
import pandas as pd
import nltk
from nltk import sent_tokenize
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as f


nltk.download('punkt')

# Configure basic logging settings
logging.basicConfig(level=logging.INFO)


def grade_bert(verbose=None):
    """This function tests the algorithm by simulating a grading procedure.
    It returns accuracy score."""

    # BERT finetuned model
    model = BertForSequenceClassification.from_pretrained("models/BERT/trained_model")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set random seed for PyTorch for results reproducibility
    torch.manual_seed(42)

    # Initialize a tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # File path of simulated student reports
    file_path = 'data/extracted_100_summaries_with_rephrase.txt'

    # Read gold labels into a pandas df
    gold = pd.read_csv('data/10_gold_phrases_and_1000_rephrases.csv', encoding='cp1252')
    gold = gold['Gold'].drop_duplicates()

    # Open the file
    with open(file_path, 'r') as file:

        # Model accuracy variable calculated by dividing total number of entailments by 100 student reports
        # and then divided by 10 in order to get a value from 0 to 1
        total_entailments = 0

        # Iterate over each line (simulated student report) in the file
        for line in file:
            entailments = 0

            # Tokenize sentences in each line
            sentences = [j.strip() for i in line.split('..') for j in sent_tokenize(i)]
            # print(len(sentences))

            # Find entailments for each gold sentence and count the total entailments found
            for gold_sentence in gold.values:

                token_ids = []
                mask_ids = []
                segments_ids = []

                # Prepare data in a format to be consumed by BERT
                hypothesis_list = sentences
                text_list = [gold_sentence] * len(hypothesis_list)

                # Iterate through above created lists in order to create token ids, segment ids and mask ids
                for (text, hypothesis) in zip(text_list, hypothesis_list):
                    # Tokenize text sentence
                    text_id = tokenizer.encode(text, add_special_tokens=False)
                    # Tokenize hypothesis sentence
                    hypothesis_id = tokenizer.encode(hypothesis, add_special_tokens=False)
                    # CLS + text + SEP + hypothesis + SEP
                    pair_token_ids = torch.tensor(
                        [tokenizer.cls_token_id] + text_id + [tokenizer.sep_token_id] + hypothesis_id + [
                            tokenizer.sep_token_id]
                    )
                    # Create segment ids (each sentence is given its id and their ids tokens are changed by ids)
                    segment_ids = torch.tensor([0] * (len(text_id) + 2) + [1] * (len(hypothesis_id) + 1))
                    # Create attention mask ids (change all sentence tokens to 1's, else to 0's)
                    attention_mask_ids = torch.tensor([1] * (len(text_id) + len(hypothesis_id) + 3))
                    # Append results to above created lists
                    token_ids.append(pair_token_ids)
                    segments_ids.append(segment_ids)
                    mask_ids.append(attention_mask_ids)

                # Pad sequences
                token_ids = pad_sequence(token_ids, batch_first=True)
                mask_ids = pad_sequence(mask_ids, batch_first=True)
                segments_ids = pad_sequence(segments_ids, batch_first=True)
                dataset = TensorDataset(token_ids, mask_ids, segments_ids)

                loader = DataLoader(
                    dataset
                )

                # Find entailments
                model.eval()
                with torch.no_grad():
                    for i, (pair_token_ids, mask_ids, seg_ids) in enumerate(loader):
                        pair_token_ids = pair_token_ids.to(device)
                        mask_ids = mask_ids.to(device)
                        seg_ids = seg_ids.to(device)

                        # Find loss and prediction values for each batch
                        prediction = model(pair_token_ids,
                                           token_type_ids=seg_ids,
                                           attention_mask=mask_ids).values()
                        prediction = list(prediction)[0]

                        # Transform prediction with log-softmax
                        prediction = f.softmax(prediction, dim=1)

                        # Does it entail or not
                        max_index = torch.argmax(prediction, dim=1).item()
                        entailment_prob = prediction[0, max_index].item()
                        if max_index == 0 and entailment_prob >= 0.95:
                            entailments += 1
                            if verbose:
                                logging.info(40 * '-')
                                logging.info(f'Prediction tensor: {prediction}')
                                logging.info(f'Golden sentence: {text_list[i]}')
                                logging.info(f'Hypothesis sentence: {hypothesis_list[i]}')

            logging.info(f'Number of entailments in one student report: {entailments}')
            if entailments > 10:
                total_entailments += 10
            else:
                total_entailments += entailments
        logging.info(total_entailments)
        accuracy = total_entailments/1000
        logging.info(accuracy)

        return accuracy
