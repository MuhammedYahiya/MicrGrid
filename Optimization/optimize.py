import tensorflow as tf
from tensorflow_model_optimization.sparsity import keras as sparsity
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

#Load the saved model
model = tf.keras.models.load_model('D:/MicroGrid/Training/best_model.h5')

#Define pruning parameters
pruning_params = {
    'pruning_schedule' : sparsity.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.5,
        begin_step=2000,
        end_step=4000
    )
}

#Apply pruning to each layer
pruned_model = tf.keras.Sequential([
    sparsity.prune_low_magnitude(layer, **pruning_params) if isinstance (layer, tf.keras.layers.Dense) else layer
    for layer in model.layers
])

#Compile the pruned model
pruned_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Prepare the dataset
data = pd.read_csv('D:/MicroGrid/Training/V2G_G2V.csv')
x = data.drop(columns=['response'])
y = data['response']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=123)

# Convert to numpy arrays (if necessary)
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Define the pruning callbacks
callbacks = [
    sparsity.UpdatePruningStep(),
    sparsity.PruningSummaries(log_dir='./logs')
]

#Train the pruned mode
pruned_model.fit(x_train, y_train, epochs=10, validation_split=0.2, callbacks=callbacks)

#Evaluate the pruned model
loss, accuracy = pruned_model.evaluate(x_test, y_test)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")

#Strip pruning wrappers
final_model = sparsity.strip_pruning(pruned_model)

#save the final pruned model
final_model.save('pruned_best_model.h5')

print("Pruned model saved as pruned_best_model.h5")