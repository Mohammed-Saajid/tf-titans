import tensorflow as tf
from tf_titans.memory import *
from tf_titans.attention import *
from tensorflow.keras.layers import Dense,Embedding

# Titans Layer (Transformer-Based Memory-Augmented Model)
class Titans(tf.keras.layers.Layer):
    """
    Implements a titans architecture with memory augmentation.
    
    This layer integrates an embedding layer, memory module, multi-head attention, 
    feed-forward network, and gating mechanisms to enhance contextual learning.
    
    Parameters:
        embedding_dim (int): Dimensionality of embeddings.
        sequence_length (int): Length of input sequences.
        num_heads (int): Number of attention heads.
        dff (int): Hidden layer size in the feed-forward network.
        total_words (int): Vocabulary size for final classification.
        rate (float): Dropout rate (default: 0.1).
        mask_zero (bool): Whether to mask zero inputs in the embedding layer (default: True).
        memory (bool): Whether to use memory augmentation (default: True).
        final_layer (bool): Whether to include a final classification layer (default: True).
        embedding_layer (bool): Whether to include an embedding layer (default: True).
        position_embedding (bool): Whether to include positional embeddings (default: True).
    """
    def __init__(self, embedding_dim, sequence_length, num_heads, dff, total_words, rate = 0.1, mask_zero=True,final_layer=True,embedding_layer=True,position_embedding=True):
        super(Titans, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dff = dff
        self.sequence_length = sequence_length
        self.total_words = total_words
        self.mask_zero = mask_zero
        self.rate = rate
        self.final_layer_ex = final_layer
        self.embedding_layer_ex = embedding_layer
        self.position_embedding_ex = position_embedding

    def build(self, input_shape):
        """
        Initializes memory, attention, feed-forward, normalization, gating, and final projection layers.
        """
        if self.embedding_layer_ex:
            self.embedding_layer = Embedding(input_dim=self.total_words, output_dim=self.embedding_dim,mask_zero=True)
        if self.position_embedding_ex:
            self.position_embedding = Embedding(input_dim=self.total_words, output_dim=self.embedding_dim)
        
        self.memory = Memory(self.embedding_dim, self.sequence_length)
        self.mha = MultiHeadAttention(self.embedding_dim * 3, self.num_heads)
        self.ffn = PositionwiseFeedforward(self.embedding_dim * 3, self.dff)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(self.rate)
        self.gate = tf.keras.layers.Dense(units=self.embedding_dim * 3, activation='sigmoid')
        self.modulation_layer = tf.keras.layers.Dense(units=self.embedding_dim * 3)
        self.memory_projection = tf.keras.layers.Dense(units = self.embedding_dim, activation="tanh")
        # Softmax layer for final classification
        if self.final_layer_ex:
            self.final_layer = tf.keras.layers.Dense(units=self.total_words, activation='softmax')    
        
        super().build(input_shape)  # Register trainable weights

    def call(self, x, mask):
        """
        Forward pass through the Titans layer.
        
        Parameters:
            x (Tensor): Input tensor.
            mask (Tensor): Mask tensor for attention.
        
        Returns:
            Tensor: Final output with softmax probabilities over vocabulary.
        """
        # Embedding and positional encoding
        if self.embedding_layer_ex:
            x = self.embedding_layer(x)
        
        if self.position_embedding_ex:
            positions = tf.range(start=0, limit=self.sequence_length, delta=1)
            position_embeddings = self.position_embedding(positions)
            x = tf.add(x,position_embeddings)

        # Memory augmentation
        memory_output = self.memory(x)
        
        # Multi-head self-attention
        attn_output = self.mha(memory_output, memory_output, memory_output, mask)
        
        # Position-wise feedforward network
        ffn_output = self.ffn(attn_output)
        
        # Layer normalization and dropout
        layer_normalization = self.layernorm(ffn_output)
        dropout = self.dropout(layer_normalization)
        
        # Skip connection
        skip = tf.add(memory_output, dropout)
        
        # Gating mechanism
        linear_gating = self.gate(skip)
        modulated_output = self.modulation_layer(linear_gating)
        gated_output = tf.multiply(linear_gating, modulated_output)
        
        # Compute attention over sequence and update memory
        attention_weights = tf.nn.softmax(tf.reduce_mean(gated_output, axis=-1), axis=1)
        weighted_memory = tf.reduce_sum(tf.multiply(gated_output, tf.expand_dims(attention_weights, axis=-1)), axis=1)
        memory_update = tf.reduce_mean(weighted_memory, axis=0, keepdims=True)
        memory_update = self.memory_projection(memory_update)

        # Update memory state through feedfoward network
        self.memory.LM.memory_state.assign(memory_update)
        
        # Memory-modulated output
        output = tf.multiply(gated_output, self.memory(x))
        
        # Final classification layer
        final_output = self.final_layer(output)
        
        return final_output
