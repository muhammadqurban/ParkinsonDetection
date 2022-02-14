from keras import layers
from keras import models
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications.resnet50 import preprocess_input
# from keras.utils import to_categorical
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np


img_path = "38 no.jpg"

img = image.load_img(img_path, target_size=(200, 200))
img_array = image.img_to_array(img)
img_array = img_array.astype("float") / 255.0
img_batch = np.expand_dims(img_array, axis=0)
    
plt.imshow(img)
plt.show()



json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model.h5")
print("Loaded model from disk")
classes =['Yes, PARKINSONS EXISTS','No, PARKINSINS DOESNOT EXIST']
## predict class index
classIndex = int(model.predict_classes(img_batch))
classIndex=classes[classIndex]
print(classIndex)


# pred = model.predict(img_batch)
# pred = pred.argmax(axis=1)[0]

# print(pred)

# converter = lite.tocoConverter.from_keras_model_file(keras_file)
# tflite_model = converter.convert()
# open("linear.tflite", "wb").write(tflite._model)