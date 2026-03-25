.PHONY: install format train eval update-branch

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

update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git add -A
	git commit -m "Update with new results" || echo "No changes to commit"
	git push --force origin HEAD:update

hf-login:
	git fetch origin update
	git switch --track -C update origin/update
	python -m pip install -r requirements.txt
	python -m pip install -U "huggingface_hub[cli]"
	hf auth login --token $(HF) --add-to-git-credential

push-hub:
	python train.py
	hf upload yuvankk/Drug-Classification ./App/app.py app.py --repo-type=space --commit-message="Sync app entry file"
	hf upload yuvankk/Drug-Classification ./App/README.md README.md --repo-type=space --commit-message="Sync Space metadata"
	hf upload yuvankk/Drug-Classification ./App/requirements.txt requirements.txt --repo-type=space --commit-message="Sync Space requirements"
	hf upload yuvankk/Drug-Classification ./model /Model --repo-type=space --commit-message="Sync Model"
	hf upload yuvankk/Drug-Classification ./result /Metrics --repo-type=space --commit-message="Sync Metrics"

deploy: hf-login push-hub