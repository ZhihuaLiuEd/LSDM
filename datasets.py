from __future__ import absolute_import, print_function

import os
import glob
import numpy as np
import cv2
import six
from torch.utils.data import Dataset

class CLUST2D(object):
    r"""`GOT-10K <http://got-10k.aitestunion.com//>`_ Dataset.

    Publication:
        ``GOT-10k: A Large High-Diversity Benchmark for Generic Object
        Tracking in the Wild``, L. Huang, X. Zhao and K. Huang, ArXiv 2018.

    Args:
        root_dir (string): Root directory of dataset where ``train``,
            ``val`` and ``test`` folders exist.
        subset (string, optional): Specify ``train``, ``val`` or ``test``
            subset of GOT-10k.
        return_meta (string, optional): If True, returns ``meta``
            of each sequence in ``__getitem__`` function, otherwise
            only returns ``img_files`` and ``anno``.
    """

    def __init__(self, root_dir, subset='Training', dim='2D', return_meta=False):
        super(CLUST2D, self).__init__()
        assert subset in ['Training', 'Test'], 'Unknown subset.'

        self.root_dir = root_dir
        self.subset = subset
        self.dim = dim

        list_file = os.path.join(root_dir, subset, dim, 'list.txt')
        self.anno_files = []
        self.data_files = []
        for line in open(list_file):
            anno_file = line.strip().split(',')[0]
            data_file = line.strip().split(',')[1]
            self.anno_files.append(anno_file)
            self.data_files.append(data_file)

    def __getitem__(self, index):
        r"""
        Args:
            index (integer or string): Index or name of a sequence.

        Returns:
            tuple: (img_files, anno) if ``return_meta`` is False, otherwise
                (img_files, anno, meta), where ``img_files`` is a list of
                file names, ``anno`` is a N x 4 (rectangles) numpy array, while
                ``meta`` is a dict contains meta information about the sequence.
        """

        img_files = sorted(glob.glob(os.path.join(self.data_files[index], '*.png')))
        anno = np.loadtxt(self.anno_files[index], delimiter=',')

        if self.subset == 'Test' and anno.ndim == 1:
            assert len(anno) == 4
            anno = anno[np.newaxis, :]

        return img_files, anno

    def __len__(self):
        return len(self.anno_files)

class Pair(Dataset):

    def __init__(self, seqs, transforms, pairs_per_seq=1):
        super(Pair, self).__init__()
        self.seqs = seqs #CLUST2D Dataset
        self.transforms = transforms
        self.pairs_per_seq = pairs_per_seq  
        self.indices = np.random.permutation(len(seqs))

    def __getitem__(self, index):

        index = self.indices[index % len(self.indices)]
        img_files, anno = self.seqs[index][:2]

        val_indices = list(n for n in range(len(anno)))

        if len(val_indices) < 2:
            index = np.random.choice(len(self))
            return self.__getitem__(index)

        # sample a frame pair
        rand_num = self._sample_one(val_indices)[0]
        target_indice = int(anno[rand_num, 0])
        long_indice = int(target_indice - 1 if target_indice - 1 > 0 else target_indice)
        init_indice = 1

        t_frame = cv2.imread(img_files[target_indice-1], cv2.IMREAD_GRAYSCALE)
        t1_frame = cv2.imread(img_files[long_indice-1], cv2.IMREAD_GRAYSCALE)
        init_frame = cv2.imread(img_files[init_indice-1], cv2.IMREAD_GRAYSCALE)

        t_box = anno[rand_num, 1:]
        init_box = anno[init_indice, 1:]

        item = (t_frame, t1_frame, init_frame, t_box, init_box)

        if self.transforms is not None:
            item = self.transforms(*item)

        return item

    def __len__(self):
        return len(self.indices) * self.pairs_per_seq

    def _sample_pair(self, indices):
        n = len(indices)
        assert n > 0

        if n == 1:
            return indices[0], indices[0]
        elif n == 2:
            return indices[0], indices[1]
        else:
            for i in range(100):
                rand_z, rand_x = np.sort(
                    np.random.choice(indices, 2, replace=False))
                if rand_x - rand_z < 100:
                    break
            else:
                rand_z = np.random.choice(indices)
                rand_x = rand_z

            return rand_z, rand_x

    def _sample_one(self, indices):
        return np.random.choice(indices, 1, replace=False)

    def cvtColor(image):
        if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
            return image
        else:
            image = image.convert('RGB')
            return image

class CLUSTDataset_Test(Dataset):
    def __init__(self, per_line, transforms):
        self.transforms = transforms
        self.anno_txt = per_line.strip().split(',')[0]
        self.data_txt = per_line.strip().split(',')[1]

        self.datas = sorted(glob.glob(os.path.join(self.data_txt, '*.png')))
        self.annos = np.loadtxt(self.anno_txt, delimiter=',')

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        init_indice = 1
        t_indice = index
        t1_indice = int(t_indice - 1 if t_indice - 1 > 0 else t_indice)

        init_frame = cv2.imread(self.datas[init_indice - 1], cv2.IMREAD_GRAYSCALE)
        t1_frame = cv2.imread(self.datas[t1_indice - 1], cv2.IMREAD_GRAYSCALE)
        t_frame = cv2.imread(self.datas[t_indice - 1], cv2.IMREAD_GRAYSCALE)

        box_init = self.annos[1:]


        item = (init_frame, t1_frame, t_frame, box_init)
        if self.transforms is not None:
            newitem = self.transforms(*item)
            return newitem
        else:
            return item

class CLUSTDataset(Dataset):
    def __init__(self, per_line, transforms=None):
        super(CLUSTDataset, self).__init__()
        self.transforms=transforms
        self.anno_txt = per_line.strip().split(',')[0]
        self.data_txt = per_line.strip().split(',')[1]

        self.datas = sorted(glob.glob(os.path.join(self.data_txt, '*.png')))
        self.annos = np.loadtxt(self.anno_txt, delimiter=',')

    def __len__(self):
        return int(len(self.annos)/4)

    def __getitem__(self, index):
        current = 1
        t = int(self.annos[index, 0])
        t1 = int(self.annos[index - 1, 0]) if index - 1 >= 0 else int(self.annos[index, 0])

        init_frame = cv2.imread(self.datas[current - 1], cv2.IMREAD_GRAYSCALE)
        t_frame = cv2.imread(self.datas[t - 1], cv2.IMREAD_GRAYSCALE)
        t1_frame = cv2.imread(self.datas[t1 - 1], cv2.IMREAD_GRAYSCALE)

        box_t = self.annos[index, 1:]
        box_t1 = self.annos[index-1, 1:] if index - 1 >= 0 else self.annos[index, 1:]

        item = (init_frame, t1_frame, t_frame, box_t1, box_t)

        if self.transforms is not None:
            new_item = self.transforms(*item)
            return new_item
        else:
            return item