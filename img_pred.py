from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from keras.applications import imagenet_utils

input_shape = (224,224)



class Imagepredict:    
    # model = None
    
    # def load_model():
    #     model = tf.keras.applications.MobileNetV2(weights="imagenet")
    #     # print("Model loaded")
    #     return model
    
    def read_image(image_encoded):
        img = Image.open(BytesIO(image_encoded))
        img = img.convert("RGB")
        return img

    def preprocess(image: Image.Image):
        image = image.resize((224,224))
        image = np.asfarray(image)
        image = image/127.5 - 1.0
        image = np.expand_dims(image, 0)
        
        return image



    # model = load_model()

    def predict(image: np.ndarray):
        
        model = tf.keras.applications.MobileNetV2((224,224,3))
        predictions = model.predict(image)
        predictions = imagenet_utils.decode_predictions(predictions)[0][0][1]
        return predictions


    
    