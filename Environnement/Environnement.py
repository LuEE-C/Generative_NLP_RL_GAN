import numpy as np
import numba
from Environnement import data_util

class Environnement:
    def __init__(self, cutoff=4, min_frequency_words=75000):
        self.ind_to_word, self.datas = data_util.convert_text_to_nptensor(cutoff=cutoff, min_frequency_words=min_frequency_words)
        self.different_words = len(self.ind_to_word)
        self.index = 0

    @numba.jit
    def query_state(self, batch_size):

        state = self.datas[self.index: self.index + batch_size]
        self.index += batch_size
        # End of epoch, shuffle dataset for next epoch
        if self.index + batch_size >= self.datas.shape[0]:
            self.index = 0
            np.random.shuffle(self.datas)
            return state, True
        else:
            return state, False

if __name__ == '__main__':
    env = Environnement()
    env.query_state(2)