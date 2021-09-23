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

def sequence_generator(data, lookback, delay, min_index, max_index=None,
              shuffle=False, batch_size=128, step=1):
    """从数据data中，截取lookback长度作为样本，总共拼凑batch_size个样本
    
    shuffle = true 打乱数据，用于提取 train数据，比较均衡。
    shuffle = false 顺序提取数据，可以用于validation 和 test数据生成，没必要打乱数据进行验证
    第一次 i = min_index + lookback ;随着generator的调用，i+= len(row) {可能是 batch_size} 不断的顺序取data中的数据
        
    data:[[1,1],[2,2],[3,3]]
    lookback: 样本的长度， ( /step后才是真实长度 )
    delay:预测的多少(延迟后)的目标
    
    返回:
    samples = [ [[1,1],[2,2],lookback // step 长 ]...batch_size 个样本 ]
    targets = []
    
    """
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows),
                            lookback // step,
                            data.shape[-1]))
        # 一维
        #targets = np.zeros((len(rows),))
        # 7维
        targets = np.zeros((len(rows),7))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            # 未来一个
            # targets[j] = data[rows[j] + delay][0]
            # 未来7个平均
            # targets[j] = data[rows[j]:rows[j]+7].mean(0)[0]
            targets[j] = data[rows[j]:rows[j]+7][:,0]
        yield samples, targets

class SequenceGenerator(object):
    def __init__(self,data,lookback,delay,batch_size):
        """
        Args:
        data:数据
        lookback:前向影响的数据
        delay:影响后续数据
        """
        self.data=data
        self.delay = delay
        self.lookback = lookback
        self.batch_size=batch_size
    
    def shuffleGenerator(self):
        max_index = len(self.data) - self.delay - 1
        while 1:
            rows = np.random.randint(
                self.lookback, max_index, size=self.batch_size)  
            samples = np.zeros((self.batch_size,
                                self.lookback ,
                                self.data.shape[-1]))  
            
            #targets = np.zeros((len(rows),7))
            for j, row in enumerate(rows):
                indices = range(rows[j] - self.lookback, rows[j], 1)
                samples[j] = self.data[indices]
                # 未来一个
                # targets[j] = data[rows[j] + delay][0]
                # 未来7个平均
                # targets[j] = data[rows[j]:rows[j]+7].mean(0)[0]
                #targets[j] = data[rows[j]:rows[j]+7][:,0]
                
            targets = self.getTarget(rows)
            yield samples, targets
    
    def orderGenerator(self):
        i = self.lookback
    
    def getTarget(self,rows):
        """
        Args:
        rows:最后元素索引数组。[100,105],
        """
        targets = np.zeros((len(rows),7))
        for j, row in enumerate(rows):
            index = rows[j]
            targets[j] = self.data[index:index+7][:,0]
        return targets
        

if __name__ =='__main__':
    aa = lstm_sequencefy(np.array([1,2,3,4,5]),2)
    print(aa)