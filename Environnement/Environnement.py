import numpy as np
import pandas as pd
import numba
from Environnement import data_util


"""
This class represents the Environnement to train on the forex data.

It formats the data, then returns the normalized input and non normalized reward state
composed of all the data that could be needed to build the actual reward, high low close

Most of the transformation are compiled using numba for speed

Lots of preprocessing (takes a few mins total)
"""
class Environnement:
    def __init__(self, cutoff=4):
        self.ind_to_word, self.datas = data_util.convert_text_to_nptensor(cutoff=cutoff)
        self.index = 0

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
    pass