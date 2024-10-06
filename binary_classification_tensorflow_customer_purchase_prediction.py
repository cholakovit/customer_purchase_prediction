import tensorflow as tf # used for building neural network
import pandas as pd # used to handle the dataset
from sklearn.model_selection import train_test_split # used for splitting the dataset into training and testing sets
from sklearn.preprocessing import StandardScaler # used to normalize the features ofr better neural network performance
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# simple dataset (fictinal)
data = {
    'age': [25, 32, 47, 51, 62, 23, 56, 45, 36, 28],
    'salary': [50000, 60000, 120000, 90000, 150000, 45000, 130000, 100000, 85000, 62000],
    'years_of_membership': [1, 4, 9, 7, 11, 2, 10, 8, 5, 3],
    'interest_in_product': [1, 1, 1, 0, 0, 1, 1, 0, 0, 1],
    'purchased': [0, 1, 1, 0, 1, 0, 1, 0, 0, 1]  # Target variable (0 = No, 1 = Yes)
}

df = pd.DataFrame(data)

x = df.drop('purchased', axis=1)
y = df['purchased']

# split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# normalaize the feature values
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

#build a simple neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, input_shape=(4,), activation='relu'),  # Hidden layer with 8 neurons
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer with 1 neuron (sigmoid for binary classification)
])

#compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# train the model
history = model.fit(x_train_scaled, y_train, epochs=50, batch_size=2, verbose=1)

# evaluate the model on the test data
test_loss, test_acc = model.evaluate(x_test_scaled, y_test, verbose=1)
print(f'\nTest Accuracy: {test_acc:.2f}')

# predict the probablity of purchasing for the test data
predictions = model.predict(x_test_scaled)

# convert probabilities to binary outputs (0 or 1)
predictions_binary = [1 if prob > 0.5 else 0 for prob in predictions]

print("\nPredictions vs Actuals:")
for i in range(len(predictions_binary)):
    print(f"Predicted: {predictions_binary[i]}, Actual: {y_test.iloc[i]}")
















