import tensorflow as tf

def train_memory(memory, gradients, alpha, theta):
    """
    Custom memory update function using a modified gradient descent approach.
    
    Parameters:
        memory (tf.keras.layers.Layer): Memory module containing trainable variables.
        gradients (list): Computed gradients for the memory variables.
        alpha (float): Weight decay parameter controlling past memory retention.
        theta (float): Suprise rate parameter for memory updates.
    """
    for var, grad in zip(memory.trainable_variables, gradients):
        if grad is not None:  # Ensure gradient exists before applying update
            update = (1 - alpha) * var - theta * grad  # Custom weight update rule
            var.assign(update)  # Assign the new updated value to the variable

def compute_loss(labels, predictions, loss_object):
    """
    Computes the loss for training.
    
    Parameters:
        labels (Tensor): True labels for the data.
        predictions (Tensor): Model predictions.
        loss_object (tf.keras.losses.Loss): Loss function to compute error.
    
    Returns:
        Tensor: Scalar loss value.
    """
    last_token_predictions = predictions[:, -1, :]  # Extract last token predictions
    y_true = tf.cast(labels, dtype=tf.int32)  # Ensure labels are of integer type
    loss = loss_object(y_true, last_token_predictions)  # Compute loss
    return tf.reduce_mean(loss)  # Return mean loss across batch

def train(model, input_data, target_data, batch_size, loss_fn, optimizer, metrics, epochs):
    """
    Training function for the model.
    
    Parameters:
        model (tf.keras.Model): The model to be trained.
        input_data (Tensor): Input training data.
        target_data (Tensor): Corresponding target labels.
        batch_size (int): Number of samples per batch.
        loss_fn (tf.keras.losses.Loss): Loss function.
        optimizer (tf.keras.optimizers.Optimizer): Optimizer for training.
        metrics (list): List of metrics for evaluation.
        epochs (int): Number of training epochs.
    
    Returns:
        None
    """
    dataset = tf.data.Dataset.from_tensor_slices((input_data, target_data)).batch(batch_size)

    for epoch in range(epochs):
        epoch_loss = 0
        if metrics is not None: 
            for metric in metrics:
                metric.reset_state()  # Reset metrics at the beginning of each epoch
        for X, y in dataset:
            with tf.GradientTape() as tape:
                predictions = model(X)
                loss = compute_loss(y, predictions, loss_fn)
                epoch_loss += loss

            gradients = tape.gradient(loss, model.trainable_variables)

            # Separate memory and non-memory variables
            memory_vars = set(id(v) for v in model.titans.memory.trainable_variables)
            memory_gradients, memory_variables = [], []
            non_memory_gradients, non_memory_variables = [], []

            for grad, var in zip(gradients, model.trainable_variables):
                if grad is not None:  # Filter out None gradients
                    if id(var) in memory_vars:
                        memory_gradients.append(grad)
                        memory_variables.append(var)
                    else:
                        non_memory_gradients.append(grad)
                        non_memory_variables.append(var)

            # Apply optimizer updates to non-memory parameters
            optimizer.apply_gradients(zip(non_memory_gradients, non_memory_variables))

            # Manually update memory variables using the custom memory update function
            train_memory(model.titans.memory, memory_gradients, alpha=0.1, theta=0.1)
            
            if metrics is not None:
                # Update metrics
                for metric in metrics:
                    metric.update_state(y, predictions[:, -1, :])  # Use only last token predictions


        avg_loss = epoch_loss / batch_size  # Compute average loss per epoch
        print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")
        # Print metric values after each epoch
        if metrics is not None:
            for metric in metrics:
                print(f"{metric.name}: {metric.result().numpy():.4f}")
    return
