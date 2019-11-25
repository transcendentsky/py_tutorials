from .models import tester

class Trainer(object):
    def __init__(self):
        self.tester = tester.Tester()

    def eee(self):
        self.tester.__print__()