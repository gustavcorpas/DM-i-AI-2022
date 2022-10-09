import random
from fastapi import APIRouter
from models.dtos import SentimentAnalysisRequestDto, SentimentAnalysisResponseDto
import numpy as np
import pandas as pd
import pickle5 as pickle

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

max_length = 500
trunc_type='post'
padding_type='post'

model = tf.keras.models.load_model('./my_model')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

router = APIRouter()


@router.post('/predict', response_model=SentimentAnalysisResponseDto)
def predict_endpoint(request: SentimentAnalysisRequestDto):

    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    prediction = model.predict(padded)
    predictions = prediction.flatten()

    predictions_mapped = []

    avarages = [0.38921918346116186, 0.5389167061687385, 0.8577790810169774, 0.9667601554368332, 0.9872908218650921]

    for p in predictions:
        if p < avarages[0] + (avarages[1] - avarages[0]) / 2:
            predictions_mapped.append(1)
        elif p < avarages[1] + (avarages[2] - avarages[1]) / 2:
            predictions_mapped.append(2)
        elif p < avarages[2] + (avarages[3] - avarages[2]) / 2:
            predictions_mapped.append(3)
        elif p < avarages[3] + (avarages[4] - avarages[3]) / 2:
            predictions_mapped.append(4)
        else:
            predictions_mapped.append(5)

    response = SentimentAnalysisResponseDto(
        scores=predictions_mapped
    )

    return response
