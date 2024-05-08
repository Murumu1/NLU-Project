import pandas as pd
import sklearn.metrics
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Concatenate, Dense, Dropout
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score
import tensorflow as tf
import numpy as np


def read_data_from_csv(csv_file):
    # Read the CSV file using pandas
    df = pd.read_csv(csv_file)

    # Extract claim, evidence, and labels
    claims = df['Claim'].tolist()  # Assuming 'claim' is the column name for claims
    evidence = df['Evidence'].tolist()  # Assuming 'evidence' is the column name for evidence
    labels = df['label'].tolist()  # Assuming 'label' is the column name for labels

    return claims, evidence, labels

# Function to preprocess claims and evidence
def preprocess_data(claims, evidence, max_seq_length):
    # Tokenize and preprocess claims
    all_text = claims + evidence
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_text)
    claims_seq = tokenizer.texts_to_sequences(claims)
    evidence_seq = tokenizer.texts_to_sequences(evidence)
    # Pad sequences to ensure uniform length
    padded_claims = pad_sequences(claims_seq, maxlen=max_seq_length, padding='post')
    padded_evidence = pad_sequences(evidence_seq, maxlen=max_seq_length, padding='post')
    return padded_claims, padded_evidence, tokenizer.word_index


# Example usage:
csv_file = 'training_data/training_data/ED/train.csv'
claims, evidence, labels = read_data_from_csv(csv_file)

# Define maximum sequence length (adjust as needed)
max_seq_length = 100  # Adjust as needed

# Preprocess the data
preprocessed_claims, preprocessed_evidence, word_index = preprocess_data(claims, evidence, max_seq_length)


# Define Siamese Bi-LSTM model
def siamese_bilstm(max_seq_length, vocab_size, embedding_dim):
    # Input layers for claims and evidence
    claim_input = Input(shape=(max_seq_length,))
    evidence_input = Input(shape=(max_seq_length,))

    # Shared embedding layer
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)

    # Embedding lookup for claims and evidence
    embedded_claims = embedding_layer(claim_input)
    embedded_evidence = embedding_layer(evidence_input)

    # Shared Bi-LSTM layer
    bilstm_layer = Bidirectional(LSTM(units=128, return_sequences=False))

    # Bi-LSTM layer for claims
    claims_bilstm = bilstm_layer(embedded_claims)
    # Bi-LSTM layer for evidence
    evidence_bilstm = bilstm_layer(embedded_evidence)

    dropout_layer = Dropout(rate=0.5)

    claims_dropout = dropout_layer(claims_bilstm)
    evidence_dropout = dropout_layer(evidence_bilstm)

    # Concatenate the output of Bi-LSTM layers
    merged = Concatenate(axis=-1)([claims_dropout, evidence_dropout])

    # Dense layer for classification
    dense_layer = Dense(units=64, activation='relu')
    output = dense_layer(merged)

    # Output layer
    output_layer = Dense(units=1, activation='sigmoid')
    predictions = output_layer(output)

    # Define model
    model = Model(inputs=[claim_input, evidence_input], outputs=predictions)
    return model


# Example usage:
max_seq_length = 100  # Define maximum sequence length
vocab_size = len(word_index) + 1  # Vocabulary size (add 1 for padding token)
embedding_dim = 256  # Dimension of word embeddings

# Create Siamese Bi-LSTM model
model = siamese_bilstm(max_seq_length, vocab_size, embedding_dim)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Example claims and evidence data (after preprocessing)
# Replace these with your actual preprocessed claims and evidence data
claims_data = preprocessed_claims  # Example claims data for testing (adjust as needed)
evidence_data = preprocessed_evidence  # Example evidence data for testing (adjust as needed)
labels_data = labels

# Convert lists to numpy arrays
claims_data = np.array(claims_data)
evidence_data = np.array(evidence_data)
labels_data = np.array(labels_data).reshape(-1, 1)

print(labels_data.shape)

# Define batch size and number of epochs
batch_size = 32
epochs = 10

# Train the model
model.fit(x=[claims_data, evidence_data], y=labels_data, batch_size=batch_size, epochs=epochs, validation_split=0.2)

# Example usage:
csv_file = 'training_data/training_data/ED/dev.csv'
d_claims, d_evidence, d_labels = read_data_from_csv(csv_file)

# Preprocess the data
preprocessed_claims, preprocessed_evidence, word_index = preprocess_data(d_claims, d_evidence, max_seq_length)

d_claims_data = preprocessed_claims  # Example claims data for testing (adjust as needed)
d_evidence_data = preprocessed_evidence  # Example evidence data for testing (adjust as needed)

# Test the model
predictions = model.predict([d_claims_data, d_evidence_data])

# Convert predictions to binary labels
binary_predictions = (predictions > 0.5).astype(int)

# Print predictions
for i, pred in enumerate(binary_predictions):
    print("Claim:", d_claims[i])
    print("Evidence:", d_evidence[i])
    print("Predicted Label:", "Supports" if pred == 1 else "Does not support")
    print("Correctly Identified?:", "Yes" if pred == d_labels[i] else "No")
    print("")

print("Accuracy:", accuracy_score(d_labels, binary_predictions))
