train: EXPERIMENT_LABEL = local
train: BRANCH = $(shell git rev-parse --abbrev-ref HEAD)
train: COMMIT = $(shell git rev-parse HEAD)
train: MODEL = resnet50/retinaface
train: CUDA_LAUNCH_BLOCKING=1
train:
	python wasp/retinaface/train.py
