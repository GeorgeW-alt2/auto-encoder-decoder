import json
import numpy as np
import tensorflow as tf
import os
from transformers import GPT2Tokenizer

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Set pad token as eos token
tokenizer.pad_token = tokenizer.eos_token

# Hyperparameters
class HParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def default_hparams():
    return HParams(
        n_vocab=50257, n_ctx=1024, n_embd=768,
        n_head=12, n_layer=12, n_positions=1024,
        batch_size=8, learning_rate=5e-5, epochs=3
    )

# GPT-2 Model Definition
class GPT2Model(tf.keras.Model):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.embedding = tf.keras.layers.Embedding(hparams.n_vocab, hparams.n_embd)
        self.lstm = tf.keras.layers.LSTM(hparams.n_embd, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(hparams.n_vocab)

    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        x, _, _ = self.lstm(x)
        logits = self.dense(x)
        return logits

# Prepare Dataset
def tokenize_text(text, max_length=128):
    tokens = tokenizer.encode(text, max_length=max_length, truncation=True, padding="max_length")
    
    return np.array(tokens)

def load_dataset(file_path, hparams):
    """Loads text data and tokenizes it."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.read().split(".")[:60]  # Limiting to 600 sentences
    
    # Tokenize and pad to fixed length
    tokenized_inputs = [tokenize_text(line.strip(), hparams.n_ctx) for line in lines]
    
    # Pad all sequences to the same length (max_length = hparams.n_ctx)
    padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_inputs, padding='post', maxlen=hparams.n_ctx, value=tokenizer.pad_token_id
    )
    
    # Create a TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices(padded_inputs)
    dataset = dataset.batch(hparams.batch_size).shuffle(1000).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


# Training Function
def train_model(model, dataset, hparams):
    optimizer = tf.keras.optimizers.Adam(learning_rate=hparams.learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    for epoch in range(hparams.epochs):
        print(f"Epoch {epoch + 1}/{hparams.epochs}")
        for step, batch in enumerate(dataset):
            with tf.GradientTape() as tape:
                logits = model(batch, training=True)
                loss = loss_fn(batch[:, 1:], logits[:, :-1])  # Shift for teacher forcing
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            print(f"Step {step}: Loss = {loss.numpy():.4f}")

    print("Training complete!")

# Generate Text
def generate_text(prompt, model, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors="tf")
    
    for _ in range(max_length):
        logits = model(input_ids)
        next_token = tf.argmax(logits[:, -1, :], axis=-1)
        input_ids = tf.concat([input_ids, next_token[:, None]], axis=-1)
    
    return tokenizer.decode(input_ids.numpy()[0])

# Main Execution
if __name__ == "__main__":
    hparams = default_hparams()
    model = GPT2Model(hparams)
    
    # Load dataset (replace 'data.txt' with your actual dataset file)
    dataset = load_dataset("test.txt", hparams)

    # Train the model
    train_model(model, dataset, hparams)

    # Test generation
    prompt = input("\nEnter a prompt: ")
    print("\nGenerated Text:\n", generate_text(prompt, model))
