from keras.models import load_model
import time
import numpy as np
import cv2
from PIL import Image, ImageFilter

class DoodlePredictor():
    def __init__(self, rest_interval=10):
        self.rest_interval = rest_interval
        self.model = load_model("../Classifier/predictor.h5")
        self.last_predict = time.time()
        self.categories = ["Apple", "Flower", "Cake", "Fish", "Star"]

    # normalize doodle image
    def normalize(self, data):
        return np.interp(data, [0, 255], [-1, 1])

    def denormalize(self, data):
        return np.interp(data, [-1, 1], [0, 255])

    # Predict drawing
    def predict(self, img_canvas):
        time_since_predict = time.time() - self.last_predict
        if time_since_predict < self.rest_interval:
            return
        
        self.last_predict = time.time()
        img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
        img_28x28 = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA) # Resize the cropped image to 28x28 pixels
        kernel = np.array([[-1,-1,-1], 
                       [-1, 9,-1],
                       [-1,-1,-1]])
        img_28x28 = cv2.filter2D(img_28x28, -1, kernel)
        # img_28x28 = Image.fromarray(img_28x28)
        # img_28x28 = img_28x28.filter(ImageFilter.SHARPEN) # sharpen image

        img_28x28 = np.array(img_28x28)
        self.visualize(img_28x28)
        input_image = img_28x28.reshape(1, 28, 28)
        input_image = self.normalize(input_image)  # normalize
        prediction = self.model.predict(input_image)
        # print("Prediction", prediction[0])
        return self.categories[np.argmax(prediction[0])]

    # Visualize a 2D array as an Image 
    def visualize(self, array):
        array = np.reshape(array, (28,28))
        img = Image.fromarray(array)
        img.show(title="Visulizing array")


        