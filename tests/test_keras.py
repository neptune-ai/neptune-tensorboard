from datetime import datetime

import neptune
import tensorflow as tf

from neptune_tensorboard import enable_tensorboard_logging_ctx


def test_keras():
    with neptune.Run() as run:
        with enable_tensorboard_logging_ctx(run):

            mnist = tf.keras.datasets.mnist

            n_samples = 5
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            x_train, x_test = x_train[:n_samples] / 255.0, x_test[:n_samples] / 255.0
            y_train, y_test = y_train[:n_samples], y_test[:n_samples]

            def create_model():
                return tf.keras.models.Sequential(
                    [
                        tf.keras.layers.Flatten(input_shape=(28, 28), name="layers_flatten"),
                        tf.keras.layers.Dense(512, activation="relu", name="layers_dense"),
                        tf.keras.layers.Dropout(0.2, name="layers_dropout"),
                        tf.keras.layers.Dense(10, activation="softmax", name="layers_dense_2"),
                    ]
                )

            model = create_model()
            model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

            log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

            model.fit(x=x_train, y=y_train, epochs=5, callbacks=[tensorboard_callback])

            run.sync()
            assert run.exists("tensorboard")
            assert run.exists("tensorboard/scalar/batch_loss")
            assert run.exists("tensorboard/scalar/batch_accuracy")
