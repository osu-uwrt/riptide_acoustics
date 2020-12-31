import numpy as np
import math

def calculate_error(a, b):
    cos = np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)
    if cos > 1:
        cos = 1
    angle_radians = math.acos(cos)
    return  angle_radians / math.pi * 180