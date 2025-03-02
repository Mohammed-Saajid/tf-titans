import tensorflow as tf
from tensorflow.keras.layers import Dense


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Implements a Multi-Head Attention mechanism with optional convolutional layers.
    
    Parameters:
        embedding_dim (int): Dimensionality of input embeddings.
        num_heads (int): Number of attention heads.
        kernel_size (int): Size of the convolutional kernel (default=1, meaning no convolutional effect).
    """
    def __init__(self, embedding_dim, num_heads, kernel_size=1):
        super(MultiHeadAttention, self).__init__()
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.depth = self.embedding_dim // self.num_heads  # Depth per attention head
        self.kernel_size = kernel_size
    
    def build(self, input_shape):
        """
        Initialize trainable layers.
        """
        # Linear transformation layers for query, key, and value projections
        self.wq = Dense(self.embedding_dim, activation=tf.keras.activations.silu)
        self.wk = Dense(self.embedding_dim, activation=tf.keras.activations.silu)
        self.wv = Dense(self.embedding_dim, activation=tf.keras.activations.silu)
        
        # Convolutional layers for additional feature extraction
        self.conv_q = tf.keras.layers.SeparableConv1D(
            filters=self.embedding_dim, kernel_size=self.kernel_size, activation=tf.keras.activations.silu, padding='same')
        self.conv_k = tf.keras.layers.SeparableConv1D(
            filters=self.embedding_dim, kernel_size=self.kernel_size, activation=tf.keras.activations.silu, padding='same')
        self.conv_v = tf.keras.layers.SeparableConv1D(
            filters=self.embedding_dim, kernel_size=self.kernel_size, activation=tf.keras.activations.silu, padding='same')
        
        # Output dense layer
        self.dense = Dense(self.embedding_dim, activation=tf.keras.activations.silu)
    
    def split_heads(self, x):
        """
        Splits the last dimension of x into (num_heads, depth) and transposes the result to shape (batch, num_heads, seq_len, depth).
        """
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        """
        Forward pass for Multi-Head Attention with convolutional enhancements.
        
        Parameters:
            v (Tensor): Value tensor.
            k (Tensor): Key tensor.
            q (Tensor): Query tensor.
            mask (Tensor): Optional mask to prevent attending to certain positions.
        
        Returns:
            Tensor: The output of the attention layer.
        """
        # Apply linear projections
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # Apply convolutional transformations
        q_conv = self.conv_q(q)
        k_conv = self.conv_k(k)
        v_conv = self.conv_v(v)
        
        # Split into multiple heads
        q = self.split_heads(q_conv)
        k = self.split_heads(k_conv)
        v = self.split_heads(v_conv)
        
        # Compute scaled dot-product attention
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        
        # Reshape output back to original dimensions
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (tf.shape(q)[0], -1, self.embedding_dim))
        
        # Final dense layer projection
        output = self.dense(concat_attention)
        return output

    def scaled_dot_product_attention(self, q, k, v, mask):
        """
        Computes scaled dot-product attention.
        
        Parameters:
            q (Tensor): Query tensor.
            k (Tensor): Key tensor.
            v (Tensor): Value tensor.
            mask (Tensor): Optional mask tensor.
        
        Returns:
            Tuple[Tensor, Tensor]: Attention output and attention weights.
        """
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)  # Large negative values to mask positions
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output, attention_weights


class PositionwiseFeedforward(tf.keras.layers.Layer):
    """
    Implements a position-wise feedforward network used in transformer architectures.
    
    Parameters:
        embedding_dim (int): Dimensionality of the input and output.
        dff (int): Dimensionality of the intermediate layer.
    """
    def __init__(self, embedding_dim, dff):
        super(PositionwiseFeedforward, self).__init__()
        self.embedding_dim = embedding_dim
        self.dff = dff
        
        # Two dense layers: one expands the dimensions, the other projects back to the original size
        self.dense1 = Dense(dff, activation='relu')  # Expansion layer
        self.dense2 = Dense(embedding_dim)  # Projection layer
    
    def call(self, x):
        """
        Forward pass through the feedforward network.
        
        Parameters:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: Transformed tensor.
        """
        x = self.dense1(x)  # Expansion
        x = self.dense2(x)  # Projection
        return x
