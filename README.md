# An Open Deep Learning Benchmark

Preview at http://dlmark.org.s3-website-us-west-2.amazonaws.com/

## How to use

1. Open an Ubuntu 16.04 instance with GPU on EC2
1. Git clone this repo
1. Install drivers `bash scripts/ubuntu16-cuda90-conda.sh`
1. Run a benchmark `./benchmark envs/mxnet-v130-cu90.yml gluoncv/detection/ssd.py`
1. Run a benchmark `./benchmark envs/mxnet-v130-cu90.yml gluoncv/detection/yolo.py`
1. Run a benchmark `./benchmark envs/mxnet-v130-cu90.yml gluoncv/detection/faster_rcnn.py`
1. Publish results `bash build/build.sh`
