import random
from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
import fruitsnuts_data
import cv2



fruits_nuts_metadata = MetadataCatalog.get("fruits_nuts")
print(fruits_nuts_metadata)
dataset_dicts = DatasetCatalog.get("fruits_nuts")


for d in random.sample(dataset_dicts, 6):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=fruits_nuts_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    print(vis.get_image()[:, :, ::-1])
    img = vis.get_image()[:, :, ::-1]
    cv2.imshow('rr', img)
    cv2.waitKey(0)