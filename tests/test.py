from datetime import datetime

import neptune
import tensorflow as tf

from neptune_tensorboard import enable_tensorboard_logging


def test_logging():
    run = neptune.init_run()
    enable_tensorboard_logging(run)

    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()

    # Scalar
    tf.summary.scalar("learning rate", data=0.1, step=1)

    # Image
    tf.summary.image("zeros", data=tf.zeros([3, 2, 2, 3]), step=1)

    # Text
    tf.summary.text("some_text", data="Hello World!", step=1)

    run.exists("tensorboard")
    run.exists("tensorboard/scalar/learning rate")
    run.exists("tensorboard/image/zeros")
    run.exists("tensorboard/text/some_text")
