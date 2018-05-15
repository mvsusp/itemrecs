import numpy as np
import os
import tensorflow as tf
from tensorflow.python.estimator.export.export import build_raw_serving_input_receiver_fn
from tensorflow.python.estimator.export.export_output import PredictOutput

tf.logging.set_verbosity(tf.logging.DEBUG)

SIGNATURE_NAME = "serving_default"
LEARNING_RATE = 0.001
USER_EMBEDDING_TENSOR_NAME = "user_embedding"
ITEM_EMBEDDING_TENSOR_NAME = "item_embedding"

def model_fn(features, labels, mode, params):
    embedding_size = 30

    user_embedding = tf.keras.layers.Embedding(
        output_dim=embedding_size,
        input_dim=57018,
        input_length=1,
        name='user',
    )(features[USER_EMBEDDING_TENSOR_NAME])

    item_embedding = tf.keras.layers.Embedding(
        output_dim=embedding_size,
        input_dim=296556,
        input_length=1,
        name='item',
    )(features[ITEM_EMBEDDING_TENSOR_NAME])

    user_vecs = tf.keras.layers.Reshape([embedding_size])(user_embedding)
    item_vecs = tf.keras.layers.Reshape([embedding_size])(item_embedding)

    y = tf.keras.layers.Dot(1, normalize=False)([user_vecs, item_vecs])

    predictions = tf.reshape(y, [-1])

    if mode == tf.estimator.ModeKeys.PREDICT:
        user = features[USER_EMBEDDING_TENSOR_NAME]
        item = features[ITEM_EMBEDDING_TENSOR_NAME]

        current_item = tf.nn.embedding_lookup(params=item_vecs, ids=item)

        tf.Print(current_item, [current_item], 'tfprint')
        tf.Print(item_vecs, [item_vecs], 'tfprint')
        tf.Print(current_item - item_vecs, [current_item - item_vecs], 'tfprint')


        distance = tf.norm(current_item - item_vecs, ord='euclidean')

        tf.Print(distance, [distance], 'tfprint')


        predictions = tf.nn.top_k(distance, 100, sorted=True)

        predictions = predictions.indices

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"products": predictions},
            export_outputs={SIGNATURE_NAME: PredictOutput({"products": predictions})}
        )

    # 2. Define the loss function for training/evaluation using Tensorflow.
    loss = tf.losses.mean_squared_error(labels, predictions)

    # 3. Define the training operation/optimizer using Tensorflow operation/optimizer.
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.001,
        optimizer="Adam",
    )
    #
    # 4. Generate predictions as Tensorflow tensors.
    predictions_dict = {"predictions": predictions}

    # 5. Generate necessary evaluation metrics.
    # Calculate root mean squared error as additional eval metric
    eval_metric_ops = {
        "mse": tf.metrics.mean_squared_error(tf.cast(labels, tf.float32), predictions)
    }

    # import pdb; pdb.set_trace()

    # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops
    )


def serving_input_fn(params):
    user = tf.placeholder(tf.int64, shape=[1])
    item = tf.placeholder(tf.int64, shape=[1])
    return build_raw_serving_input_receiver_fn({
        USER_EMBEDDING_TENSOR_NAME: user, ITEM_EMBEDDING_TENSOR_NAME: item
    })()


params = {"learning_rate": LEARNING_RATE}


def train_input_fn(training_dir, params):
    return _input_fn(training_dir, 'train.csv')


def eval_input_fn(training_dir, params):
    return _input_fn(training_dir, 'test.csv')


def _input_fn(training_dir, training_filename):
    training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=os.path.join(training_dir, training_filename),
        target_dtype=np.int,
        features_dtype=np.int
    )



    print(training_set.data)

    return tf.estimator.inputs.numpy_input_fn(
        x={
            USER_EMBEDDING_TENSOR_NAME: np.array(training_set.data[:, 0]),
            ITEM_EMBEDDING_TENSOR_NAME: np.array(training_set.data[:, 1])
        },


        y=training_set.target,
        shuffle=True,
        batch_size=64,
        num_epochs=None
    )()
