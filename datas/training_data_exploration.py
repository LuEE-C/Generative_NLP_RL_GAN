import numpy as np
import Environnement.data_util as data_util

ind_to_word, datas = data_util.convert_text_to_nptensor(cutoff=6, min_frequency_words=75000)
