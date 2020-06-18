import warnings
import functools

class deprecated:
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""

    def __init__(self, message):
        self.message = message

    def __call__(self, func):
      @functools.wraps(func)
      def new_func(*args, **kwargs):
          message = "Call to deprecated function {}. {}".format(func.__name__, self.message)
          warnings.simplefilter('always', DeprecationWarning)  # turn off filter
          warnings.warn(message,
                        category=DeprecationWarning,
                        stacklevel=2)
          warnings.simplefilter('default', DeprecationWarning)  # reset filter
          return func(*args, **kwargs)
      return new_func
      
    