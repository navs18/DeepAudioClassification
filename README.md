Audio Data Processing & Preprocessing: Bird audio clips are loaded, downsampled, converted into spectrograms, and used as input for training a deep learning model to detect Capuchinbird calls.
CNN Model Training: A convolutional neural network (CNN) is built and trained to classify audio segments as either Capuchinbird calls or not, using TensorFlow datasets for efficient data handling.
Prediction & Result Export: The trained model predicts bird calls in long audio recordings, groups consecutive detections, and exports the results to a CSV file for further analysis.
