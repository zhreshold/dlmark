#!/bin/bash

# generating benchmark results
python3 ~/dlmark/gluoncv/classification/cnn.py
python3 ~/dlmark/gluoncv/detection/ssd.py
python3 ~/dlmark/gluoncv/detection/faster_rcnn.py
python3 ~/dlmark/gluoncv/detection/yolo.py
python3 ~/dlmark/gluoncv/detection/center_net.py
python3 ~/dlmark/gluoncv/semantic_segmentation/semantic_segmentation.py
python3 ~/dlmark/gluoncv/pose/pose.py

# generating embedded htmls
python3 ~/dlmark/gluoncv/classification/cnn_genhtml.py
python3 ~/dlmark/gluoncv/detection/det_genhtml.py
python3 ~/dlmark/gluoncv/semantic_segmentation/semantic_segmentation_genhtml.py
python3 ~/dlmark/gluoncv/pose/pose_genhtml.py

# upload to s3
aws s3 cp ~/dlmark/gluoncv/classification/classification_throughputs.html s3://zhiz-cache/gluoncv-throughput-benchmarks/ --acl public-read
aws s3 cp ~/dlmark/gluoncv/detection/detection_throughputs.html s3://zhiz-cache/gluoncv-throughput-benchmarks/ --acl public-read
aws s3 cp ~/dlmark/gluoncv/detection/detection_coco_per_class.html s3://zhiz-cache/gluoncv-throughput-benchmarks/ --acl public-read
aws s3 cp ~/dlmark/gluoncv/semantic_segmentation/semantic_segmentation_throughputs.html s3://zhiz-cache/gluoncv-throughput-benchmarks/ --acl public-read
aws s3 cp ~/dlmark/gluoncv/pose/pose_throughputs.html s3://zhiz-cache/gluoncv-throughput-benchmarks/ --acl public-read
