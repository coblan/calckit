import numpy as np

def to_one_hot(labels, dimension=46):
    """
    labels=[1,2,3,4]
    """
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

def vectorize_sequences(sequences, dimension=10):
    """
    sequences=np.array([1,2,3])   一维 与one_hot输出一致
    sequences=np.array([[1,2,2],[2,3,3]])  输出 2x10 的矩阵，index有值的地方为1
    
    array([[0., 1., 1., ..., 0., 0., 0.],
       [0., 0., 1., ..., 0., 0., 0.]])
       
    """
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results