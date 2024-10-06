import os
import cv2
import imghdr
import numpy as np
from matplotlib import pyplot as plt
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.metrics import Precision, Recall, BinaryAccuracy
import numpy as np
from matplotlib import pyplot as plt

# Set the visible GPU devices to GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf

# Avoid out-of-memory (OOM) errors by setting GPU memory consumption growth
# This allows TensorFlow to gradually increase GPU memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Define data directory and allowed image extensions
data_dir = 'data'
image_exts = ['jpeg', 'jpg', 'bmp', 'png']

# Iterate through the dataset and verify the image extensions are valid
for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            # Read the image to check if it's valid
            img = cv2.imread(image_path)
            # Check the file type of the image
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                # If the file type is not in the allowed extensions, remove the file
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            # If there's an issue with reading the image, print the issue
            print('Issue with image {}'.format(image_path))

# Load the dataset from the directory using TensorFlow's utility function
data = tf.keras.utils.image_dataset_from_directory('data')

# Create an iterator to visualize a batch of data
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

# Plot the first 4 images from the batch along with their labels
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])
plt.show()

# Normalize the images by dividing the pixel values by 255
data = data.map(lambda x, y: (x / 255, y))

# Convert data to a numpy iterator to confirm normalization
data.as_numpy_iterator().next()

# Split the data into training, validation, and test sets
train_size = int(len(data) * .7)
val_size = int(len(data) * .2) + 1
test_size = int(len(data) * .1) + 1

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

# Build the Convolutional Neural Network (CNN) model
model = Sequential()

# Add convolutional and pooling layers to the model
model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

# Compile the model with Adam optimizer and binary cross-entropy loss
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

# Print the model summary to understand its structure
model.summary()

# Define log directory for TensorBoard
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# Train the model using the training data, validating with the validation data
hist = model.fit(train, epochs=100, validation_data=val, callbacks=[tensorboard_callback])

# Initialize precision, recall, and accuracy metrics
pre = Precision()
re = Recall()
acc = BinaryAccuracy()

# Evaluate the model using the test data
for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

# Print the precision, recall, and accuracy of the model
print(pre.result(), re.result(), acc.result())

# Read an image for prediction
img = cv2.imread('cat-3535404_1280.jpg')

# Resize the image to match the input size of the model
resize = tf.image.resize(img, (256, 256))

# Make a prediction using the model
yhat = model.predict(np.expand_dims(resize / 255, 0))

# Print the prediction result
if yhat < 0.5:
  print(f"the model predicted that the image is a { os.listdir(data_dir)[0]}")
else:
  print(f"the model predicted that the image is a { os.listdir(data_dir)[1]}")

print(yhat)