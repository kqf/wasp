train: EXPERIMENT_LABEL = local
train: BRANCH = $(shell git rev-parse --abbrev-ref HEAD)
train: COMMIT = $(shell git rev-parse HEAD)
train: model = resnet50/retinaface
train:
	python letracking/train.py
