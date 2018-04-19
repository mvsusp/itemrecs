import logging
import sys
import os

import boto3
import botocore
import sagemaker
from sagemaker.tensorflow import TensorFlow

import sagemaker
import tensorflow as tf
import numpy as np


tf_estimator = TensorFlow(entry_point='itemembd.py', role='SageMakerRole',
                          training_steps=10, evaluation_steps=None,
                          train_instance_count=1, train_instance_type='ml.p2.xlarge')

tf_estimator.fit('s3://bucket/path/to/training/data')

predictor = tf_estimator.deploy()

user = tf.make_tensor_proto(values=np.asarray([10]), shape=[1], dtype=tf.float64)
item = tf.make_tensor_proto(values=np.asarray([10]), shape=[1], dtype=tf.float64)

# not working
predictor.predict({'user': 10, 'item': 10})

predictor.predict({'user': [10], 'item': [10]})


d = {'user': user, 'item': item}

predictor.predict(d)

# working
predictor.predict(item)