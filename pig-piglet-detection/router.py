
import random
import base64
import numpy as np
from typing import List
from loguru import logger
from fastapi import APIRouter
from models.dtos import PredictRequestDto, PredictResponseDto, BoundingBoxClassification

import matplotlib.pyplot as plt
from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageOps

import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import tensorflow_hub as hub
import os
import base64    


model_url = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
detector = hub.Module(model_url)

router = APIRouter()

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

@router.post('/predict', response_model=PredictResponseDto)
def predict_endpoint(request: PredictRequestDto):
    
    encoded_img: str = request.img
    image_string = base64.b64decode(encoded_img) 

    # Run the graph we just created
    result_out, image_out = sess.run(
        [detector_output, decoded_image],
        feed_dict={image_string_placeholder: image_string}
    )


    response = []

    for index, element in enumerate(result_out["detection_class_entities"]):
        el = str(element)
        if el == "b'Pig'":
            random_class = random.randint(0, 1)  # 0 = PIG, 1 = PIGLET
            response.append(BoundingBoxClassification(
                class_id=random_class,
                min_x=result_out["detection_boxes"][index][0],
                min_y=result_out["detection_boxes"][index][1],
                max_x=result_out["detection_boxes"][index][2],
                max_y=result_out["detection_boxes"][index][3],
                confidence=result_out["detection_scores"][index])
    )

    print(response)

    return response


def predict(img: np.ndarray) -> List[BoundingBoxClassification]:
    logger.info(f'Recieved image: {img.shape}')
    bounding_boxes: List[BoundingBoxClassification] = []




    for _ in range(random.randint(0, 9)):
        bounding_box: BoundingBoxClassification = get_dummy_box()
        bounding_boxes.append(bounding_box)
        logger.info(bounding_box)
    return bounding_boxes


def get_dummy_box() -> BoundingBoxClassification:
    random_class = random.randint(0, 1)  # 0 = PIG, 1 = PIGLET
    random_min_x = random.uniform(0, .9)
    random_min_y = random.uniform(0, .9)
    random_max_x = random.uniform(random_min_x + .05, 1)
    random_max_y = random.uniform(random_min_y + .05, 1)
    return BoundingBoxClassification(
        class_id=random_class,
        min_x=random_min_x,
        min_y=random_min_y,
        max_x=random_max_x,
        max_y=random_max_y,
        confidence=random.uniform(0, 1)
    )
