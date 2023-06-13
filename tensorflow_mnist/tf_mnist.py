import os
import cv2
import numpy as np
import tensorflow as tf
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def train_model(epochs):
    # Training data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    # Train Model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs)
    model.save('mnist.model')


def evaluate_img(num):
    model = tf.keras.models.load_model('mnist.model')

    while os.path.isfile(f"sample_data/{num}.png"):
        img = cv2.imread(f"sample_data/{num}.png")[:, :, 0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This is a picture of the number: {np.argmax(prediction)}")
        num = input("Enter a number 0-9: ")
    print(f"Image 'sample_data/{num}.png' not found!")


if __name__ == '__main__':
    # train_model(10)
    evaluate_img(input("Enter a number 0-9: "))
