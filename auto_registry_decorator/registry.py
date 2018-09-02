
_MODELS = {}


def register_model(name=None):
  """Register a model. name defaults to class name snake-cased."""
  print("[###] Register model: {}".format(name))
  '''
  def decorator(model_cls, registration_name=None):
    """Registers & returns model_cls with registration_name or default name."""
    model_name = registration_name or default_name(model_cls)
    if model_name in _MODELS and not tf.contrib.eager.in_eager_mode():
      raise LookupError("Model %s already registered." % model_name)
    model_cls.REGISTERED_NAME = model_name
    _MODELS[model_name] = model_cls
    return model_cls

  # Handle if decorator was used without parens
  if callable(name):
    model_cls = name
    return decorator(model_cls, registration_name=default_name(model_cls))

  return lambda model_cls: decorator(model_cls, name)
  '''
  model = name
  if callable(name):
    print("[#] Callable!")

  return model