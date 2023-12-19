all: wider/train.json wider/valid.json wider/test.json
images: WIDER_train.zip
labels: retinaface_gt_v1.1.zip

wider/%.json: wider-labels
	python /convert.py \
		--dataset wider-labels/$*/label.txt \
		--ofile $@

wider-labels: $(labels)
	mkdir -p $@
	unzip $^ -d wider-labels
	mv wider-labels/val wider-labels/valid

$(labels):
	gdown 1BbXxIiY-F74SumCNG6iwmJJ5K3heoemT

wider: $(images)
	unzip $^
	mv WIDER_train $@

$(images):
	gdown 15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M
