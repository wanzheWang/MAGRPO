from tensorboard.compat import tensorflow_stub as _tf_stub

# Expose TensorFlow-stub API so TensorBoard can read event files.
globals().update(_tf_stub.__dict__)
