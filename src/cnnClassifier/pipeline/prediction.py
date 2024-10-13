import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array  # Updated imports for image loading
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
    
    def predict(self):
        # Load the trained model
        model = load_model(os.path.join("artifacts", "training", "model.h5"))

        # Load and preprocess the image
        imagename = self.filename
        test_image = load_img(imagename, target_size=(224, 224))  # Updated method for loading the image
        test_image = img_to_array(test_image)                     # Convert image to array
        test_image = np.expand_dims(test_image, axis=0)           # Expand dims to make it batch size
        test_image = test_image / 255.0                           # Normalize pixel values (0-1)

        # Predict using the model
        prediction = model.predict(test_image)                    # Get prediction probabilities
        result = np.argmax(prediction, axis=1)                    # Get index of the highest probability class

        # Output the prediction based on the result
        if result[0] == 1:
            return [{"image": "Tumor"}]
        else:
            return [{"image": "Normal"}]

