from experiments.learn import memory

class MemoryHead():
    def __init__(self, memory_size = 32768, vocabulary_size = 2):
        self.memory_size = memory_size
        self.vocabulary_size = vocabulary_size
        self.mem = None

    def query(self, intended_output = None):
        return self.mem.query(self.features, intended_output)

    def policy_model(self, data, feature_tensor, intended_output):
        self.features = feature_tensor
        self.intended_output = intended_output
        if self.mem is None:
            self.mem = memory.Memory(feature_tensor.shape[1], self.memory_size, self.vocabulary_size)#, var_cache_device="/cpu:0", nn_device="/cpu:0"
        return self.query(intended_output)
