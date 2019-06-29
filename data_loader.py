from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import sys
import pickle
import numpy as np
import lmdb
import torch
from tqdm import tqdm


def default_loader(path):
    try:
        im = Image.open(path).convert('RGB')
        return im
    except:
        print(..., file=sys.stderr)
        return Image.new('RGB', (224, 224), 'white')


class ImagerLoader(data.Dataset):
    def __init__(self, img_path, transform=None, target_transform=None,
                 loader=default_loader, square=False, data_path=None, partition=None, sem_reg=None):

        if data_path == None:
            raise Exception('No data path specified.')

        if partition is None:
            raise Exception('Unknown partition type %s.' % partition)
        else:
            self.partition = partition

        self.env = lmdb.open(os.path.join(data_path, partition + '_lmdb'), max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)

        with open(os.path.join(data_path, partition + '_keys.pkl'), 'rb') as f:
            self.ids = pickle.load(f)

        self.square = square
        self.imgPath = img_path
        self.mismtch = 0.8
        self.maxInst = 20

        if sem_reg is not None:
            self.semantic_reg = sem_reg
        else:
            self.semantic_reg = False

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        #print ("xianlaiyibofuck")
        recipId = self.ids[index]
        # we force 80 percent of them to be a mismatch
        if self.partition == 'train':
            match = np.random.uniform() > self.mismtch
        elif self.partition == 'val' or self.partition == 'test':
            match = True
        else:
            raise ("Partition name not well defined")

        target = match and 1 or -1

        with self.env.begin(write=False) as txn:
            serialized_sample = txn.get(self.ids[index].encode())
        sample = pickle.loads(serialized_sample,encoding="bytes")
        imgs = sample['imgs'.encode()]
        #print (imgs)
        # image
        #print ("caonima")
        if target == 1:
            if self.partition == 'train':
                # We do only use the first five images per recipe during training
                imgIdx = np.random.choice(range(min(5, len(imgs))))
            else:
                imgIdx = 0


            loader_path = [imgs[imgIdx]['id'.encode()].decode()[i] for i in range(4)]
            #print (loader_path)
            loader_path = os.path.join(*loader_path)
            #print (loader_path)
            path = os.path.join(self.imgPath, 'recipe1M_images_'+self.partition, self.partition, loader_path, imgs[imgIdx]['id'.encode()].decode())

        else:
            # we randomly pick one non-matching image
            all_idx = range(len(self.ids))
            rndindex = np.random.choice(all_idx)
            while rndindex == index:
                rndindex = np.random.choice(all_idx)  # pick a random index

            with self.env.begin(write=False) as txn:
                serialized_sample = txn.get(self.ids[rndindex].encode())

            rndsample = pickle.loads(serialized_sample,encoding="bytes")
            rndimgs = rndsample['imgs'.encode()]

            if self.partition == 'train':  # if training we pick a random image
                # We do only use the first five images per recipe during training
                imgIdx = np.random.choice(range(min(5, len(rndimgs))))
            else:
                imgIdx = 0

            loader_path = [rndimgs[imgIdx]['id'.encode()].decode()[i] for i in range(4)]
            # print (loader_path)
            loader_path = os.path.join(*loader_path)
            # print (loader_path)
            path = os.path.join(self.imgPath,'recipe1M_images_'+self.partition, self.partition, loader_path, rndimgs[imgIdx]['id'.encode()].decode())

            # instructions
        #print ("rilegou")

        instrs = sample['intrs'.encode()]
        itr_ln = len(instrs)
        t_inst = np.zeros((self.maxInst, np.shape(instrs)[1]), dtype=np.float32)
        t_inst[:itr_ln][:] = instrs
        instrs = torch.FloatTensor(t_inst)

        # ingredients
        ingrs = sample['ingrs'.encode()].astype(int)
        ingrs = torch.LongTensor(ingrs)
        igr_ln = max(np.nonzero(sample['ingrs'.encode()])[0]) + 1
        #print ("malegebi")
        # load image

        img = self.loader(path)
        #print (img)

        if self.square:
            img = img.resize(self.square)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        rec_class = sample['classes'.encode()] - 1
        rec_id = self.ids[index]

        if target == -1:
            img_class = rndsample['classes'.encode()] - 1
            img_id = self.ids[rndindex]
        else:
            img_class = sample['classes'.encode()] - 1
            img_id = self.ids[index]

        #print (img)
        # output
        if self.partition == 'train':
            if self.semantic_reg:
                return [img, instrs, itr_ln, ingrs, igr_ln], [target, img_class, rec_class]
            else:
                return [img, instrs, itr_ln, ingrs, igr_ln], [target]
        else:
            if self.semantic_reg:
                return [img, instrs, itr_ln, ingrs, igr_ln], [target, img_class, rec_class, img_id, rec_id]
            else:
                return [img, instrs, itr_ln, ingrs, igr_ln], [target, img_id, rec_id]

    def __len__(self):
        return len(self.ids)
