import pandas as pd
import tensorflow as tf

""" Data Preparation. """
data = pd.read_csv("train.csv", index_col=0)

FEATURES = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "GarageCars"]
y = data.SalePrice
X = data[FEATURES]


""" Modeling """
model = tf.keras.Sequential([
    tf.keras.layers.Dense(20, activation="relu", input_shape=[len(FEATURES)]), # Neural network with 20 neurons
    tf.keras.layers.Dense(52, activation="relu"), # Neural network with 52 neurons
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(), 
    loss="mae",
    metrics=["mae", "mse"] # MAE - Mean Absolute Error
)


""" Train (aka "fit") and evaluate the Model. """
EPOCHS = 5

model.fit(
    X, # Features
    y, # Labels
    validation_split=0.2, # Set aside 20% of our data to validate our model
    epochs=EPOCHS
)

""" Testing """
test_data = pd.read_csv("test.csv", index_col=0)

test_X = test_data[FEATURES]
predictions = model.predict(test_X)

HOUSE = 0

print(test_X.iloc[HOUSE])
print("\nPredicted Price:", predictions[HOUSE])