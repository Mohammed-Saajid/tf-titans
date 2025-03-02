import tensorflow as tf
from tensorflow.keras.layers import Dense

# Long Term Memory Layer
class LongTermMemory(tf.keras.layers.Layer):
    """
    Implements a simple long-term memory layer that retains learned information.
    
    Parameters:
        units (int): Number of units in the layer.
        activation (function): Activation function to use (default: SiLU).
    """
    def __init__(self, units, activation=tf.keras.activations.silu):
        super(LongTermMemory, self).__init__()
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        """
        Initializes dense layers and a memory state.
        """
        self.fc1 = Dense(self.units, activation=self.activation, input_shape=input_shape)
        self.fc2 = Dense(self.units * 2, activation=self.activation)
        self.fc3 = Dense(self.units, activation=self.activation)
        
        # Test time Trainable memory state
        self.memory_state = self.add_weight(
            shape=(1, self.units),  # Global memory state
            initializer="zeros",
            trainable=True  # Allows updates via feedforward and backpropagation
        )
    
    def call(self, inputs):
        """
        Forward pass of the Long-Term Memory layer.
        
        Parameters:
            inputs (Tensor): Input tensor.
        
        Returns:
            Tensor: Processed output incorporating memory state.
        """
        
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        x = tf.multiply(x, self.memory_state)  # Element-wise multiplication with memory state
        
        return x


# Complete Memory Implementation
class Memory(tf.keras.layers.Layer):
    """
    Implements a memory mechanism combining a long-term memory layer with persistent memory.
    
    Parameters:
        embedding_dim (int): Dimensionality of embeddings.
        sequence_length (int): Length of the input sequence.
        activation (function): Activation function to use (default: SiLU).
    """
    def __init__(self, embedding_dim, sequence_length, activation=tf.keras.activations.silu):
        super(Memory, self).__init__()
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.activation = activation

    def build(self, input_shape):
        """
        Initializes layers for memory processing.
        """
        # Query transformation layer
        self.LMWq = Dense(units=self.embedding_dim, activation=self.activation, use_bias=False)
        
        # Long-term memory module
        self.LM = LongTermMemory(self.embedding_dim, activation=self.activation)
        
        # Persistent memory layer
        self.persistent_memory = Dense(self.embedding_dim)

    def call(self, inputs):
        """
        Forward pass through the Memory module.
        
        Parameters:
            inputs (Tensor): Input tensor.
        
        Returns:
            Tensor: Concatenated output of transformed inputs, long-term memory, and persistent memory.
        """
        x = self.LMWq(inputs)  # Apply query transformation
        x = self.LM(x)  # Pass through long-term memory layer
        persistent_memory = self.persistent_memory(inputs)  # persistent memory
        
        # Concatenate input, long-term memory output, and persistent memory along the last axis
        return tf.concat([inputs, x, persistent_memory], axis=-1)
