from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import tf_keras as tk

np.set_printoptions(suppress=True)

model = tk.models.load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

image = Image.open("imegesd.jpg").convert("RGB")
image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

image_array = np.asarray(image)
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
data[0] = normalized_image_array

prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

print("Class:", class_name[2:].strip())
print("Confidence Score:", confidence_score)
