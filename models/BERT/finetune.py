import torch
from transformers import BertForSequenceClassification, AdamW
import time
import logging


# Configure basic logging settings
logging.basicConfig(level=logging.INFO)


class FineTune:
    """This class is used to finetune BERT model for entailment recognition."""
    def __init__(self, loader_train, loader_val, epochs=3):
        self.loader_train = loader_train
        self.loader_val = loader_val
        self.epochs = epochs

    def finetune(self):
        """This function finetunes BERT model for entailment recognition and saves it."""
        # Assign a GPU (or CPU) for training
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Create a model instance
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=3)

        # Assign the model to the device
        model.to(device)

        # Configure an optimization strategy
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, correct_bias=False)

        for epoch in range(self.epochs):
            start = time.time()
            model.train()
            train_loss_total = 0
            train_acc_total = 0
            verbose = 0
            for i, (pair_token_ids, mask_ids, segments_ids, y) in enumerate(self.loader_train):

                # Keep track
                verbose += 1
                logging.info(f'Batch number: {verbose} out of {len(self.loader_train)}')

                optimizer.zero_grad()
                pair_token_ids = pair_token_ids.to(device)
                mask_ids = mask_ids.to(device)
                segments_ids = segments_ids.to(device)
                labels = y.to(device)

                # Find loss and prediction values for each batch
                loss, prediction = model(pair_token_ids,
                                         token_type_ids=segments_ids,
                                         attention_mask=mask_ids,
                                         labels=labels).values()

                # Transform prediction with log-softmax (log for computational stability and ease of computational)
                prediction = torch.log_softmax(prediction, dim=1)

                # Calcualte accuracy (float is needed to ensure a float result when performing a division operation)
                acc = (prediction.argmax(dim=1) == labels).sum().float() / float(labels.size(0))

                # Back propagate
                loss.backward()

                # Update weights
                optimizer.step()

                train_loss_total += loss.item()
                train_acc_total += acc.item()

            # Calculate avg accuracy over all batches
            train_acc = train_acc_total / len(self.loader_train)

            # Calculate avg loss over all batches
            train_loss = train_loss_total / len(self.loader_train)

            # Evaluation stage
            model.eval()
            total_val_acc = 0
            total_val_loss = 0
            with torch.no_grad():
                for i, (pair_token_ids, mask_ids, seg_ids, y) in enumerate(self.loader_val):
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

                    total_val_loss += loss.item()
                    total_val_acc += acc.item()

            # Calculate avg accuracy over all batches
            val_acc = total_val_acc / len(self.loader_val)

            # Calculate avg loss over all batches
            val_loss = total_val_loss / len(self.loader_val)

            # Time elapsed
            end = time.time()
            hours, rem = divmod(end - start, 3600)
            minutes, seconds = divmod(rem, 60)

            logging.info(
                f'Epoch {epoch + 1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | '
                f'val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}')

            logging.info(
                "Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
            )

        model.save_pretrained("models/BERT/trained_model")

        return model
