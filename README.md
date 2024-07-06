Develop a hand gesture recognition model that can accurately identify and classify different hand gestures from image or video data, enabling intuitive human-computer interaction and gesture-based control systems using CNN.

# Hand Gesture Recognition using CNN

This project is focused on recognizing hand gestures using Convolutional Neural Networks (CNN). The project consists of two main components:



## File 1: PRODIGY_ML_04.ipynb

### Description
This Jupyter Notebook file contains the code for building a CNN model for hand gesture recognition. It starts with data preprocessing, reading gesture images, processing them, and building the CNN model.

### Key Steps:
- Reading and Preprocessing Data: Gesture images are read from the dataset, and preprocessing steps, including image resizing, are applied.
- Label Encoding: Gesture labels are converted into numerical representations.
- Data Split: The dataset is split into training and testing sets.
- CNN Model Building: A CNN model is constructed with multiple layers for feature extraction and classification.
- Model Training: The model is trained using the training data, with early stopping to prevent overfitting.
- Model Evaluation: The model's accuracy is evaluated on the testing data.
- Learning Curves: Learning curves are plotted to visualize model performance.
- Model Saving: The trained model is saved for later use.

## File 2: Real-TimeRec.ipynb

### Description
This Jupyter Notebook file contains the code for real-time hand gesture recognition. It uses the pre-trained CNN model to classify gestures in real-time using the laptop's camera. The code also utilizes KNN background subtraction to segment the hand from the frame.

### Key Steps:
- Pre-trained Model Loading: The pre-trained CNN model from HandGesRec.ipynb is loaded.
- Camera Initialization: The laptop's camera is initialized for real-time gesture recognition.
- Background Subtraction: KNN background subtraction is used to extract the hand from the frame.
- Skin Color Detection: The code detects skin color within the frame.
- Combining Masks: The skin mask is combined with the background subtraction mask.
- Morphological Operations: Morphological operations are applied to reduce noise in the segmented hand.
- Model Predictions: The segmented hand is resized, and the CNN model predicts the gesture in real-time.
- Display: The recognized gesture is displayed on the frame.

### Requirements
- Python
- OpenCV
- NumPy
- PIL (Pillow)
- Matplotlib
- Keras
- TensorFlow

### Usage
- Run HandGesRec.ipynb to train and save the CNN model.
- Run Real-TimeRec.ipynb to perform real-time hand gesture recognition.

### References
- Example Dataset: [Hand Gesture Recognition Dataset](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)
- TensorFlow Documentation: [TensorFlow Guide](https://www.tensorflow.org/guide)
- Keras Documentation: [Keras API Reference](https://keras.io/api/)

