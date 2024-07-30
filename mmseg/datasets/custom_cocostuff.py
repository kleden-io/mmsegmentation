# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class CustomCOCOStuffDataset(BaseSegDataset):
    METAINFO = dict(
        classes=(
            'person', 'ground-other'),
        palette=[[0, 192, 64], [0,0,0]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='_labelTrainIds.png',
                 reduce_zero_label = False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, 
            reduce_zero_label=reduce_zero_label,
            seg_map_suffix=seg_map_suffix, 
            **kwargs)
