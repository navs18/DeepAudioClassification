# Deep Audio Classification: Capuchinbird Call Detection

## 1. Overview
This project focuses on automated audio classification to detect **Capuchinbird calls** from extensive forest audio recordings. Using **deep learning and convolutional neural networks (CNNs)**, the model efficiently identifies and classifies bird vocalizations, reducing the need for manual analysis.

## 2. Features
- **Audio Data Processing & Preprocessing**: Loads, downsamples, and converts audio clips into spectrograms for model training.
- **Deep Learning Model (CNN)**: A convolutional neural network is trained using TensorFlow to classify audio as Capuchinbird calls or not.
- **Prediction & Result Export**: The trained model analyzes long audio recordings, groups consecutive detections, and outputs results in a CSV file for further study.

## 3. Libraries and Tools Used
- **Python** (Primary programming language)
- **TensorFlow & Keras** (Deep learning framework for CNN modeling)
- **TensorFlow I/O** (Efficient audio processing)
- **Matplotlib** (For visualization of spectrograms and training results)
- **OS & CSV** (File handling and exporting results)
- **Itertools** (For handling iterative data processing tasks)

## 4. Installation
To run this project, install the required dependencies:

```bash
pip install tensorflow tensorflow-io matplotlib
```

Ensure that you have Python 3.7+ installed.

## 5. Dataset
The dataset consists of **raw audio recordings** containing Capuchinbird calls and other forest sounds. Each audio clip is converted into a **spectrogram** before being passed to the CNN model.

## 6. Preprocessing Steps
1. **Load Audio Data**: Read and process raw audio files.
2. **Downsampling**: Convert to a lower sample rate for efficiency.
3. **Spectrogram Conversion**: Transform waveform data into spectrogram images for CNN input.
4. **Data Augmentation**: Apply transformations like noise addition or time shifts (if applicable).

## 7. Model Architecture
The **CNN model** used in this project follows these key layers:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

## 8. Training and Evaluation
- **Loss Function**: Categorical Cross-Entropy
- **Optimizer**: Adam (for faster convergence)
- **Metrics**: Accuracy, Precision, Recall
- **Performance**: Achieved **97.36% accuracy** in Capuchinbird call classification.

## 9. Running the Model
Run the following script to train the model:

```bash
python train.py
```

To make predictions on new audio files:

```bash
python predict.py --input path_to_audio_file
```

## 10. Results and Output
- **Predictions**: The model analyzes long audio recordings and detects Capuchinbird calls.
- **CSV Export**: Results are saved in `results.csv`, grouping consecutive detections.

## 11. Future Improvements
- Enhancing model robustness with more diverse training data.
- Exploring **Recurrent Neural Networks (RNNs)** for temporal audio analysis.
- Deploying as a real-time classification API.

## 12. Contributors
- **Naveen Kumar** - [GitHub](https://github.com/navs18)

