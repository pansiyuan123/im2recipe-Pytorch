# -*- coding: utf-8 -*-
import time
from tqdm import *
import numpy as np
def readdata_fromlmdb():
    import lmdb
    import binascii
    from functools import partial
    import pickle
    pickle.load = partial(pickle.load, encoding="latin1")
    pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")

    env_db = lmdb.open("./data/test_lmdb")

    txn = env_db.begin()
    # print (txn.get(str(200)))
    # "'00003a70b1', '00047059be', '00059477e2', '0007a28fe7', '000bba053c', '000d5e4996', '00100336d5', '001243534a', '0012733d1d'"

    i = 0
    for key, value in txn.cursor():  # 遍历
        if i < 10:
            print(i, key, value)
        i += 1

    with open('./test_keys.pkl', 'rb') as f:
        ids = pickle.load(f, encoding='iso-8859-1')

    print(ids[0:10])
    serialized_sample = txn.get(ids[10].encode())
    print("fuck", serialized_sample)
    sample = pickle.loads(serialized_sample, encoding="bytes")
    print(sample)
    imgs = sample['imgs'.encode()]
    print(imgs)

    env_db.close()
