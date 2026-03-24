.PHONY: install format train eval

install:
	python -m pip install --upgrade pip &&\
		python -m pip install black &&\
		python -m pip install -r requirements.txt

format:
	python -m black *.py

train:
	python train.py

eval:
	echo "## Model Metrics" > report.md
	if [ -f ./result/metrics.txt ]; then cat ./result/metrics.txt >> report.md; else echo "Metrics file not found. Run training pipeline first." >> report.md; fi
	echo '\n## Confusion Matrix Plot' >> report.md
	if [ -f ./result/model_results.png ]; then echo '![Confusion Matrix](./result/model_results.png)' >> report.md; else echo "Confusion matrix image not found." >> report.md; fi
	cml comment create report.md