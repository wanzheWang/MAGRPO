try:
    import tensorflow as tf
    from tensorboard.compat import tensorflow_stub as _tf_stub
    if not hasattr(tf, "io"):
        tf.io = _tf_stub.io
    if not hasattr(tf, "compat"):
        tf.compat = _tf_stub.compat
    if not hasattr(tf, "errors"):
        tf.errors = _tf_stub.errors
    if not hasattr(tf, "__version__"):
        tf.__version__ = "stub"
except Exception:
    pass
