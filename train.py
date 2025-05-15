from __future__ import absolute_import

from datasets import CLUST2D

from lsdmtrack import LSDMTracker


if __name__ == '__main__':
    list_root = 'path/to/list.txt'

    tracker = LSDMTracker()
    tracker.train_over(list_root)