import numpy as np


feature = np.load('taa/data/DAD/vgg16_features/testing/b001_000490.npz', allow_pickle=True)
print(feature.keys())
'''
data
labels
det
ID
'''

print(feature['data'].shape) # (100, 20, 4096)
print(feature['labels'].shape) # (2,)
print(feature['det'].shape) # (100, 19, 6)
print(feature['ID'].shape) # ()

print(feature['det'][0][:10])
print(feature['det'][1][:10])
print(feature['det'][2][:10])
print(feature['det'][3][:10])