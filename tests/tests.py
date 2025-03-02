import unittest
import tensorflow as tf
from tf_titans.titans import Titans
from tf_titans.train import compute_loss
from tf_titans.memory import Memory

class TestTitansModel(unittest.TestCase):
    def setUp(self):
        """Setup a basic model before each test"""
        self.embedding_dim = 128
        self.sequence_length = 40
        self.num_heads = 8
        self.dff = 256
        self.total_words = 10000
        self.model = Titans(
            embedding_dim=self.embedding_dim,
            sequence_length=self.sequence_length,
            num_heads=self.num_heads,
            dff=self.dff,
            total_words=self.total_words
        )

    def test_model_output_shape(self):
        """Test if model produces expected output shape"""
        sample_input = tf.ones((1628, self.sequence_length))
        output = self.model(sample_input, mask=None)
        self.assertEqual(output.shape, (1628, self.sequence_length, self.total_words))

    def test_memory_initialization(self):
        """Test if memory initializes properly"""
        memory = Memory(self.embedding_dim, self.sequence_length)
        self.assertIsInstance(memory, Memory)

class TestTrainingFunctions(unittest.TestCase):
    def test_compute_loss(self):
        """Test loss computation"""
        labels = tf.constant([[3]])  # Shape (1, 1) to match last_token_predictions
        predictions = tf.random.uniform((1, 4, 10000))  # Shape (1, 4, vocab_size)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        loss = compute_loss(labels, predictions, loss_fn)
        self.assertTrue(loss.numpy() >= 0)  # Loss should be non-negative

if __name__ == '__main__':
    unittest.main()
