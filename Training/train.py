import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



# Load the dataset
data = pd.read_csv("D:\MicroGrid\Training\V2G_G2V.csv")
#prepare the data for tensorflow/keras model
x = data.drop(columns=['response'])
y = data['response']


#split the data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=123)

#Define input shape and number of classes
input_shape = x_train.shape[1] #Number of features
num_classes = len(y.unique()) #Number of unique classes in target variable

#create a keras model
model = Sequential([
    Dense(64, activation='relu', input_shape=(input_shape,)),
    Dense(64,activation='relu'),
    Dense(num_classes, activation='softmax')
])

#compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Train the model
model.fit(x_train, y_train, epochs=10, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")

# Save the Keras model as .h5
model.save('best_model.h5')

print("Keras model saved as best_model.h5")