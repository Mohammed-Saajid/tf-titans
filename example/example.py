import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding
import regex as re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tf_titans.titans import Titans
from tf_titans.train import train
import numpy as np

# Step 1: Read and preprocess text data
def file_to_sentence_list(file_path):
    """
    Reads a text file and splits it into sentences.
    Args:
        file_path (str): Path to the text file.
    Returns:
        list: A list of sentences extracted from the file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    # Splitting text into sentences based on punctuation marks
    sentences = [sentence.strip() for sentence in re.split(r'(?<=[.!?])\s+', text) if sentence.strip()]
    return sentences

# Step 2: Prepare input sequences
def prepare_input_sequences(text_data, tokenizer):
    """
    Converts text data into sequences of tokens and creates n-gram sequences.
    Args:
        text_data (list): List of sentences.
        tokenizer (Tokenizer): Keras Tokenizer instance.
    Returns:
        list: A list of n-gram sequences.
    """
    input_sequences = []
    for line in text_data:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences

# Load and process text data
file_path = 'C:\\Users\\admin\\Research\\Pytorch Titans\\titans\\tf-titans\\example\\pizza.txt'
text_data = file_to_sentence_list(file_path)

# Tokenize the text data
tokenizer = Tokenizer(oov_token="<UNK>")
tokenizer.fit_on_texts(text_data)
total_words = len(tokenizer.word_index) + 1  # Vocabulary size including OOV token

# Create input sequences
input_sequences = prepare_input_sequences(text_data, tokenizer)

# Pad sequences and split into predictors (X) and labels (y)
max_length = max([len(seq) for seq in input_sequences])  # Find max sequence length
input_sequences = tf.convert_to_tensor(pad_sequences(input_sequences, maxlen=max_length+1, padding='pre'), dtype=tf.int32)
X, y = input_sequences[:, :-1], input_sequences[:, -1]  # Last token is the label

# Model parameters
embedding_dim = 128
num_heads = 8
dff = 512
batch_size = 32

# Define a custom model using Titans Transformer-based architecture
class CustomModel(tf.keras.Model):
    def __init__(self, embedding_dim, sequence_length, num_heads, dff, total_words):
        """
        Custom Transformer-based model using Titans library.
        Args:
            embedding_dim (int): Size of the word embedding.
            sequence_length (int): Maximum length of input sequences.
            num_heads (int): Number of attention heads.
            dff (int): Dimension of the feed-forward network.
            total_words (int): Vocabulary size.
        """
        super(CustomModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.num_heads = num_heads
        self.dff = dff
        self.total_words = total_words

    def build(self, input_shape):
        """
        Builds the model layers, including the Titans module.
        """
        self.titans = Titans(embedding_dim=self.embedding_dim,
                             sequence_length=self.sequence_length,
                             num_heads=self.num_heads,
                             dff=self.dff,
                             total_words=self.total_words)
        super().build(input_shape)

    def call(self, inputs):
        """
        Defines forward pass of the model.
        """
        x = self.titans(inputs, mask=None)
        return x

# Instantiate and compile the model
model = CustomModel(embedding_dim=embedding_dim, sequence_length=max_length, num_heads=num_heads, dff=dff, total_words=total_words)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.build(input_shape=X.shape)

# Train the model
train(model=model, input_data=X, target_data=y, batch_size=batch_size, 
      loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE), 
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()], epochs=10, 
      optimizer=tf.keras.optimizers.Adam())

# Generate next word predictions
def generate_text(seed_text, next_words, model, tokenizer, max_length):
    """
    Generates text using the trained model.
    Args:
        seed_text (str): Initial text prompt.
        next_words (int): Number of words to generate.
        model (tf.keras.Model): Trained model.
        tokenizer (Tokenizer): Keras tokenizer instance.
        max_length (int): Maximum sequence length.
    Returns:
        str: Generated text.
    """
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_length, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)  # Shape: (1, max_length, total_words)
        last_time_step_probs = predicted_probs[:, -1, :]  # Extract last time step probabilities
        predicted_index = np.argmax(last_time_step_probs, axis=-1).item()  # Get index of highest probability word
        predicted_word = tokenizer.index_word.get(predicted_index, "<UNK>")  # Convert index to word
        seed_text += " " + predicted_word  # Append predicted word to input text
    return seed_text

# Example usage of text generation
seed_text = "Pizza have"
next_words = 5
generated_text = generate_text(seed_text, next_words, model, tokenizer, max_length)
print("Next predicted words:", generated_text)
