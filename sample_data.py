import glob
import os
path = '/data/dataset/torcs/tmp1'

file_list = glob.glob('{}/*jpg'.format(path))

print(file_list)