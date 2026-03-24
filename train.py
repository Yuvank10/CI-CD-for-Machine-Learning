from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def main() -> None:
	data_path = Path("Data/drug.csv")
	result_dir = Path("result")
	model_dir = Path("model")
	result_dir.mkdir(parents=True, exist_ok=True)
	model_dir.mkdir(parents=True, exist_ok=True)

	drug_df = pd.read_csv(data_path)

	x = drug_df.drop(columns=["Drug"])
	y = drug_df["Drug"]

	x_train, x_test, y_train, y_test = train_test_split(
		x, y, test_size=0.3, random_state=125, stratify=y
	)

	cat_cols = ["Sex", "BP", "Cholesterol"]
	num_cols = ["Age", "Na_to_K"]

	preprocess = ColumnTransformer(
		transformers=[
			("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
			("num", "passthrough", num_cols),
		]
	)

	pipeline = Pipeline(
		steps=[
			("preprocess", preprocess),
			("model", RandomForestClassifier(n_estimators=200, random_state=125)),
		]
	)

	pipeline.fit(x_train, y_train)
	predictions = pipeline.predict(x_test)

	accuracy = accuracy_score(y_test, predictions)
	f1 = f1_score(y_test, predictions, average="weighted")

	metrics_path = result_dir / "metrics.txt"
	metrics_path.write_text(
		f"Accuracy = {accuracy:.2f}, F1 Score = {f1:.2f}.\n", encoding="utf-8"
	)

	disp = ConfusionMatrixDisplay.from_predictions(y_test, predictions)
	disp.figure_.savefig(result_dir / "model_results.png", dpi=120, bbox_inches="tight")
	plt.close(disp.figure_)

	dump(pipeline, model_dir / "drug_pipeline.joblib")

	print(f"Saved metrics: {metrics_path}")
	print(f"Saved confusion matrix: {result_dir / 'model_results.png'}")
	print(f"Saved model: {model_dir / 'drug_pipeline.joblib'}")


if __name__ == "__main__":
	main()
