import numpy as np

input = np.ones((10,10))
loc = np.random.randint(0,10, (2,))
input[loc[0], loc[1]] = 9

print(input, loc)

class LocDataLoader(object):
    def __init__(self, size=10, batch_size=32):
        self.size = 10
        self.batch_size = batch_size

    def get_item(self):
        input = np.ones((self.size,self.size))
        loc = np.random.randint(0,self.size, (2,))
        input[loc[0], loc[1]] = 9
        return input, loc

    def forward(self):
        inputs = []
        locs = []
        for i in range(self.batch_size):
            input, loc = self.get_item()
            inputs.append(input)
            locs.append(loc)
        inputs = np.array(inputs)
        locs = np.array(locs)
        inputs = np.expand_dims(inputs,-1)
        return inputs, locs

loader = LocDataLoader()
x,y = loader.forward()
print(x.shape, y.shape)