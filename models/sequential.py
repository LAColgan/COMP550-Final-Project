import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import numpy as np
from joblib import dump



max_sequence_length = 50 
embedding_dim = 100
class Sequential:
    def __init__(self) -> None:
        self.dev_df=pd.read_csv("data/snli_1.0_dev.txt", sep='\t')
        self.test_df=pd.read_csv("data/snli_1.0_test.txt", sep='\t')
        self.train_df=pd.read_csv("data/snli_1.0_train.txt", sep='\t')

        self.model=Sequential()
        self.label_encoder=LabelEncoder()
        self.tokenizer = Tokenizer()
        self.model=None

    def train(self, df): #used with the dev set for time purposes
        df['encoded_labels'] = self.label_encoder.fit_transform(df['gold_label'])

        self.tokenizer.fit_on_texts(df['sentence1'].values + df['sentence2'].values)
        vocab_size = len(self.tokenizer.word_index) + 1

        premises_input = Input(shape=(max_sequence_length,), name='premises_input')
        hypotheses_input = Input(shape=(max_sequence_length,), name='hypotheses_input')

        embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length)
        premises_embedding = embedding_layer(premises_input)
        hypotheses_embedding = embedding_layer(hypotheses_input)
        premises_sequences = self.tokenizer.texts_to_sequences(df['sentence1'].values)
        hypotheses_sequences = self.tokenizer.texts_to_sequences(df['sentence2'].values)
        premises_padded = pad_sequences(premises_sequences, maxlen=max_sequence_length)
        hypotheses_padded = pad_sequences(hypotheses_sequences, maxlen=max_sequence_length)
        labels_one_hot = to_categorical(df['encoded_labels'])

        lstm_layer = LSTM(units=100)
        premises_lstm = lstm_layer(premises_embedding)
        hypotheses_lstm = lstm_layer(hypotheses_embedding)

        merged = concatenate([premises_lstm, hypotheses_lstm])

        output_layer = Dense(units=len(self.label_encoder.classes_), activation='softmax', name='output')
        predictions = output_layer(merged)

        model = Model(inputs=[premises_input, hypotheses_input], outputs=predictions)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit({'premises_input': premises_padded, 'hypotheses_input': hypotheses_padded}, labels_one_hot, epochs=5, batch_size=32, validation_split=0.2)

        dump(model,'sequential.joblib')
        self.model=model

    def test_df(self):
        self.test_df = self.test_df[['sentence1', 'sentence2', 'gold_label']]
        self.test_df = self.test_df.replace('-', pd.NA)
        self.test_df = self.test_df.dropna()
        self.test_df['encoded_labels'] = self.label_encoder.fit_transform(self.test_df['gold_label'])


        self.tokenizer.fit_on_texts(self.test_df['sentence1'].values + self.test_df['sentence2'].values)

        test_one_hot = to_categorical(self.test_df['encoded_labels'])


        test_premises_sequences = self.tokenizer.texts_to_sequences(self.test_df['sentence1'].values)
        test_hypotheses_sequences = self.tokenizer.texts_to_sequences(self.test_df['sentence2'].values)

        test_premises_padded = pad_sequences(test_premises_sequences, maxlen=max_sequence_length)
        test_hypotheses_padded = pad_sequences(test_hypotheses_sequences, maxlen=max_sequence_length)
        loss, accuracy = self.model.evaluate([test_premises_padded, test_hypotheses_padded], test_one_hot)
        print(f"Test Accuracy: {accuracy}")


    




        