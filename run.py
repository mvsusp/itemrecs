import numpy as np
import tensorflow as tf
import sagemaker
from sagemaker.tensorflow import TensorFlow

tf_estimator = TensorFlow(entry_point='itemembd.py', role='SageMakerRole',
                          training_steps=1, evaluation_steps=1,
                          train_instance_count=1, train_instance_type='local')

sagemaker_session = sagemaker.Session()

inputs = sagemaker_session.upload_data(path='.', key_prefix='itemrecs')

tf_estimator.fit(inputs)

predictor = tf_estimator.deploy(initial_instance_count=1, instance_type='local')

user = tf.make_tensor_proto(values=np.asarray([10]), shape=[1], dtype=tf.float64)
item = tf.make_tensor_proto(values=np.asarray([10]), shape=[1], dtype=tf.float64)

# not working
predictor.predict({'user': 10, 'item': 10})

predictor.predict({'user': [10], 'item': [10]})


d = {'user': user, 'item': item}

predictor.predict(d)

# working
predictor.predict(item)
