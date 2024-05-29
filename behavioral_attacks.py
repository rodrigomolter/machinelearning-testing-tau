from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.data import Dataset
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import TextVectorization, Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

import tensorflow_datasets as tfds
import tensorflow as tf

def main():
  print('TensorFlow version:', tf.__version__)
  tfds.disable_progress_bar()


  """ Collect Data """
  train_dataset, val_dataset = load_imdb_reviews()

  print(train_dataset.element_spec)
  print("\n")

  #Peek into an actual entry of the dataset
  for example, label in train_dataset.take(1):
      print("review :", example.numpy())
      print("label  :", label.numpy())



  """ Prepare Data """

  # We pass in a vocab_size for how big we want the model's vocabulary to be.
  train_dataset, val_dataset = batch_datasets(train_dataset, val_dataset)
  encoder, vocab = create_text_encoder(train_dataset, vocab_size=1000)
  print("\n")
  print("Beginning of vocabulary:\n", vocab[:20])
  print("\nEnd of vocabulary:\n", vocab[-20:])



  """ Train the Model"""
  print("\n")
  print("Starting the training of the LSTM Model...")
  print("This will take a while (~10min), so be patient...\n")


  #Train single LSTM layer model.
  LSTM_model = build_single_LSTM_model(encoder)
  compile_model(LSTM_model)
  LSTM_history = train_model(
      LSTM_model,
      train_dataset,
      val_dataset,
      epochs=5,
      validation_steps=2
  )

  #Evaluate the model against our validation dataset.
  val_loss, val_acc = LSTM_model.evaluate(val_dataset)
  print("\n")
  print('Validation Loss:', val_loss)
  print('Validation Accuracy:', val_acc)

  plot_history(LSTM_history)


  """ Test the Model
  Remember:

    * If the prediction is >= 0, it is positive
    * If the prediction is  < 0, it is negative 
  
  """

  # Our model takes a list of texts to predict on
  sample_text = ["movie good", "movie bad"]
  predictions = LSTM_model.predict(np.array(sample_text))

  # and outputs a list with their respective scores (aka predictions)
  print(predictions)
  print(predictions >= 0)

  print("\n")
  text = input("Predict Sentiment for: ")
  output = LSTM_model.predict(np.array([text]))
  print(output)






def load_imdb_reviews() -> Tuple[Dataset, Dataset]:
  """ Load IMDB Reviews and split into train and validation datasets. """
  dataset, _ = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
  train_dataset, val_dataset = dataset["train"], dataset["test"]
  return train_dataset, val_dataset


def batch_datasets(train_dataset, val_dataset, buffer_size=10000, batch_size=64) -> Tuple[Dataset, Dataset]:
  """ Batch and prefetch train and test datasets. """
  train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
  val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
  return train_dataset, val_dataset


def create_text_encoder(train_dataset, vocab_size=1000) -> Tuple[TextVectorization, np.ndarray]:
  """ Vectorize texts and create vocabulary based on train_dataset. """
  encoder = TextVectorization(max_tokens=vocab_size)
  encoder.adapt(train_dataset.map(lambda text, label: text))
  vocab = np.array(encoder.get_vocabulary())
  return encoder, vocab


def build_single_LSTM_model(encoder, output_dim=64) -> Model:
  """ Build a sequential model with a single LSTM bidirectional layer. """
  model = Sequential(
      [
          encoder,
          Embedding(
              input_dim=len(encoder.get_vocabulary()),
              output_dim=output_dim,
              # Use masking to handle the variable sequence lengths
              mask_zero=True,
          ),
          Bidirectional(LSTM(output_dim)),
          Dense(output_dim, activation="relu"),
          Dense(1),
      ]
  )
  return model


def build_double_LSTM_model(encoder, output_dim=64) -> Model:
  """ Build a sequential model with two LSTM bidirectional layers and a Dropout convolution. """
  model = Sequential([
      encoder,
      Embedding(len(encoder.get_vocabulary()), output_dim=output_dim, mask_zero=True),
      Bidirectional(LSTM(64, return_sequences=True)),
      Bidirectional(LSTM(32)),
      Dense(64, activation="relu"),
      Dropout(0.5),
      Dense(1)
  ])
  return model


def compile_model(model):
  """ Compile the model using standard parameters. """
  model.compile(
      loss=BinaryCrossentropy(from_logits=True),
      optimizer=Adam(learning_rate=1e-4),
      metrics=["accuracy"],
  )


def train_model(model, train_dataset, val_dataset=None, epochs=5, validation_steps=0):
  """ Train the model using validation data, then return the training history for analysis. """
  history = model.fit(
      train_dataset,
      epochs=epochs,
      validation_data=val_dataset,
      validation_steps=validation_steps,
  )
  return history


def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history["val_" + metric], "")
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, "val_" + metric])


def plot_history(history):
  plt.figure(figsize=(16, 8))
  plt.subplot(1, 2, 1)
  plot_graphs(history, "accuracy")
  plt.ylim(None, 1)
  plt.subplot(1, 2, 2)
  plot_graphs(history, "loss")
  plt.ylim(0, None)




if __name__ == "__main__":
  main()