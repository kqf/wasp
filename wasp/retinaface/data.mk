all: wider/train.json wider/valid.json wider/test.json

wider/%.json: wider-labels
	python wasp/retinaface/convert.py \
		--dataset wider-labels/$*/label.txt \
		--ofile $@

wider-labels: retinaface_gt_v1.1.zip
	mkdir -p $@
	unzip $^ -d wider-labels
	mv wider-labels/val wider-labels/valid

retinaface_gt_v1.1.zip:
	gdown 1BbXxIiY-F74SumCNG6iwmJJ5K3heoemT

wider: WIDER_train.zip
	unzip $^
	mv WIDER_train $@

WIDER_train.zip:
	gdown 15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M
