import numpy as np

def distance_euclid(vec_a: np.array, vec_b: np.array):
    return np.linalg.norm(vec_b - vec_a)