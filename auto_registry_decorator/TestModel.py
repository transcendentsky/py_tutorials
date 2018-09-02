import registry

@registry.register_model
class TestModel(object):
  def __init__(self):
    print("OK")

