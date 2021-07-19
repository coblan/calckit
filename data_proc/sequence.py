import numpy as np

def lstm_sequencefy(X,time_steps):
    """分割数据
    
    X=[1,2,3,4,5]
    time_steps=2
    
    分割为: [[1,2],[2,3],[3,4],[4,5]]
    """
    data_len = len(X)
    XX=  [X[i:i+time_steps] for i in range(data_len-time_steps+1)] 
    XXX = np.reshape(XX, (data_len - time_steps+1, time_steps, -1))
    return XXX


if __name__ =='__main__':
    aa = lstm_sequencefy(np.array([1,2,3,4,5]),2)
    print(aa)