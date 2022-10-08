import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageOps
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import tensorflow_hub as hub
import os


sample_image_path = "./training/2.jpg"

with tf.Graph().as_default():
    # Create our inference graph
    image_string_placeholder = tf.placeholder(tf.string)
    decoded_image = tf.image.decode_jpeg(image_string_placeholder)
    decoded_image_float = tf.image.convert_image_dtype(
        image=decoded_image, dtype=tf.float32
    )

    # Expanding image from (height, width, 3) to (1, height, width, 3)
    image_tensor = tf.expand_dims(decoded_image_float, 0)

    # Load the model from tfhub.dev, and create a detector_output tensor
    model_url = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
    detector = hub.Module(model_url)
    detector_output = detector(image_tensor, as_dict=True)
    
    # Initialize the Session
    init_ops = [tf.global_variables_initializer(), tf.tables_initializer()]
    sess = tf.Session()
    sess.run(init_ops)


# Load our sample image into a binary string
with tf.gfile.Open(sample_image_path, "rb") as binfile:
    image_string = binfile.read()

# Run the graph we just created
result_out, image_out = sess.run(
    [detector_output, decoded_image],
    feed_dict={image_string_placeholder: image_string}
)

# result_out["detection_boxes"]
# result_out["detection_class_entities"]
# result_out["detection_scores"]

labels_arr = []
boxes_arr = []
scores_arr = []

for index, element in enumerate(result_out["detection_class_entities"]):
    el = str(element)
    if el == "b'Pig'":
        labels_arr.append(random.randint(0, 1))
        boxes_arr.append(result_out["detection_boxes"][index])
        scores_arr.append(result_out["detection_scores"][index])

print(labels_arr, boxes_arr, scores_arr)