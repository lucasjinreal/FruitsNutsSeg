"""

this script will using pycoco API
draw our converted annotation to check
if result is right or not

"""
from pycocotools.coco import COCO
import os
import sys
import cv2
from pycocotools import mask as maskUtils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import skimage.io as io


def run(coco_img_root, ann_f):
    data_dir = coco_img_root
    coco = COCO(ann_f)

    cats = coco.loadCats(coco.getCatIds())
    print('cats: {}'.format(cats))

    img_ids = coco.getImgIds()
    print('img_ids: {}'.format(img_ids))

    img = coco.loadImgs(img_ids[1])
    print('checking img: {}, id: {}'.format(img, img_ids[1]))

    img_f = os.path.join(data_dir, img[0]['file_name'])

    # draw instances
    anno_ids = coco.getAnnIds(imgIds=img[0]['id'])
    annos = coco.loadAnns(anno_ids)

    I = io.imread(img_f)
    plt.imshow(I)
    plt.axis('off')

    coco.showAnns(annos)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('call like python3 vis_coco.py path/to/img/root /path/to/anno.json')
    else:
        run(sys.argv[1], sys.argv[2])

