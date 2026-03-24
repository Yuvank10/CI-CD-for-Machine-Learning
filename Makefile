.PHONY: install format train eval

install:
	python -m pip install --upgrade pip &&\
		python -m pip install -r requirements.txt

format:
	black *.py

train:
	python train.py

eval:
	echo "## Model Metrics" > report.md
	cat ./result/metrics.txt >> report.md
	echo '\n## Confusion Matrix Plot' >> report.md
	echo '![Confusion Matrix](./result/model_results.png)' >> report.md
	cml comment create report.md