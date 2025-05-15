import os
import time
from datasets import CLUSTDataset_Test
from transforms import SiamFCTransforms_Test
from config import config

from torch.utils.data import DataLoader

class ExperimentCLUST2D(object):
    def __init__(self, list_root, model_pt, result_dir='results', report_dir='reports'):
        super(ExperimentCLUST2D, self).__init__()
        self.list_root = list_root
        self.model_pt = model_pt
        self.result_dir = result_dir
        self.report_dir = os.path.join(report_dir, 'CLUST2D')
        self.nbins_iou = 101
        self.repetitions = 3

    def run(self, tracker, visualiza=False):

        print('Running tracker %s on CLUST2D...' % tracker.name)

        # loop over the complete dataset
        transforms = SiamFCTransforms_Test(
            exemplar_sz=config.exemplar_sz,
            instance_sz=config.instance_sz,
            context=config.context)
        data = []
        for line in open(self.list_root):
            data.append(line)
        for line_iter, line in enumerate(data):
            print("Track on Patient: {}".format(line))
            anno_line = line.strip().split(",")[0]
            anno_file = anno_line.split("/")[-1]
            patient_id = anno_line.split("/")[-2]
            marker_id = anno_file.split(".")[0]
            print("marker_id :", marker_id)
            print("patient_id :", patient_id)
            clust_dataset = CLUSTDataset_Test(line, transforms=transforms)
            clust_dataloader = DataLoader(clust_dataset,
                                          batch_size=1,
                                          shuffle=False,
                                          )
            tracker.test_over(clust_dataloader, patient_id, marker_id, self.result_dir, self.report_dir)

    def report(self):
        pass

