.PHONY: install train val export infer test zip

install:
	conda env create -f environment.yml

train:
	python -m aquayolo26 train

val:
	python -m aquayolo26 val

export:
	python -m aquayolo26 export

infer:
	python -m aquayolo26 infer

test:
	python -m pytest -q

zip:
	cd outputs && zip -qr AquaYOLO26_release.zip AquaYOLO26_release

full-train:
	python -m aquayolo26.train_full
