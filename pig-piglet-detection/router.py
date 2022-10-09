
import random
import base64
from fastapi import APIRouter
from models.dtos import PredictRequestDto, PredictResponseDto, BoundingBoxClassification

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import tensorflow_hub as hub
import base64    


router = APIRouter()

print("hello from router")

with tf.Graph().as_default():
    # Create our inference graph
    image_string_placeholder = tf.placeholder(tf.string)
    decoded_image = tf.image.decode_image(
            image_string_placeholder,
            channels=3,
            expand_animations=False,
        )
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
    print(request)
    encoded_img: str = request.img
    image_string = base64.b64decode(encoded_img) 
    print("continue...")
    print(image_string)
    print("continue...")

    # Run the graph we just created
    result_out, image_out = sess.run(
        [detector_output, decoded_image],
        feed_dict={image_string_placeholder: image_string}
    )

    print(result_out["detection_class_entities"])

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
    return PredictResponseDto(
        boxes=response
    )


