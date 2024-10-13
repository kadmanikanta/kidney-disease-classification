from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline

# Environment settings
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# ClientApp class for handling the prediction pipeline
class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"  # Set the filename for the input image
        self.classifier = PredictionPipeline(self.filename)  # Initialize the prediction pipeline

# Instantiate the ClientApp globally
clApp = ClientApp()

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')  # Renders home page (make sure the template exists)

@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def trainRoute():
    # Run a DVC repro command to trigger the training pipeline
    os.system("dvc repro")
    return "Training done successfully!"

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    # Extract the image from the JSON payload
    image = request.json['image']
    
    # Decode and save the image
    decodeImage(image, clApp.filename)
    
    # Perform prediction using the prediction pipeline
    result = clApp.classifier.predict()
    
    # Return the result as a JSON response
    return jsonify(result)

if __name__ == "__main__":
    # Run the Flask app on 0.0.0.0 and port 8080 (AWS setup)
    app.run(host='0.0.0.0', port=8080)
