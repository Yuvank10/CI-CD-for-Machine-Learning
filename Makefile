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
	git commit -am "Update with new results"
	git push --force origin HEAD:update

hf-login:
    git pull origin update
    git switch update
    pip install -U "huggingface_hub[cli]"
    huggingface-cli login --token $(HF) --add-to-git-credential

push-hub:
    huggingface-cli upload kingabzpro/Drug-Classification ./App --repo-type=space --commit-message="Sync App files"
    huggingface-cli upload kingabzpro/Drug-Classification ./Model /Model --repo-type=space --commit-message="Sync Model"
    huggingface-cli upload kingabzpro/Drug-Classification ./Results /Metrics --repo-type=space --commit-message="Sync Model"

deploy: hf-login push-hub