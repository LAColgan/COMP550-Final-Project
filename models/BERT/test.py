import torch
from transformers import BertForSequenceClassification
import time
import logging

# Configure basic logging settings
logging.basicConfig(level=logging.INFO)


class Test:
    """This class is used to test a finetuned BERT model for entailment recognition."""

    def __init__(self, loader_test, model=None):
        self.loader_test = loader_test
        self.model = model

    def test(self):
        """This function tests a finetuned BERT model for entailment recognition."""
        # Assign a GPU (or CPU) for testing
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Create a model instance
        # model = self.model
        model = BertForSequenceClassification.from_pretrained("models/BERT/trained_model")

        # Assign the model to the device
        model.to(device)

        # Test stage
        model.eval()
        total_test_acc = 0
        total_test_loss = 0
        start = time.time()
        with torch.no_grad():
            for i, (pair_token_ids, mask_ids, seg_ids, y) in enumerate(self.loader_test):
                pair_token_ids = pair_token_ids.to(device)
                mask_ids = mask_ids.to(device)
                seg_ids = seg_ids.to(device)
                labels = y.to(device)

                # Find loss and prediction values for each batch
                loss, prediction = model(pair_token_ids,
                                         token_type_ids=seg_ids,
                                         attention_mask=mask_ids,
                                         labels=labels).values()

                # Transform prediction with log-softmax (log for computational stability and ease of computational)
                prediction = torch.log_softmax(prediction, dim=1)

                # Calcualte accuracy (float is needed to ensure a float result when performing a division operation)
                acc = (prediction.argmax(dim=1) == labels).sum().float() / float(labels.size(0))

                total_test_loss += loss.item()
                total_test_acc += acc.item()

        # Calculate avg accuracy over all batches
        test_acc = total_test_acc / len(self.loader_test)

        # Calculate avg loss over all batches
        test_loss = total_test_loss / len(self.loader_test)

        # Time elapsed
        end = time.time()
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)

        logging.info(f'Test_loss: {test_loss:.4f} test_acc: {test_acc:.4f}')

        logging.info(
            "Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
        )

        return test_acc, test_loss
