import gradio as gr
import pandas as pd
from joblib import load as joblib_load
import skops.io as sio
import warnings
from pathlib import Path
from sklearn.exceptions import InconsistentVersionWarning

# Suppress sklearn version warnings when loading serialized artifacts.
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

trusted_types = [
    "sklearn.pipeline.Pipeline",
    "sklearn.preprocessing.OneHotEncoder",
    "sklearn.preprocessing.StandardScaler",
    "sklearn.compose.ColumnTransformer",
    "sklearn.preprocessing.OrdinalEncoder",
    "sklearn.impute.SimpleImputer",
    "sklearn.tree.DecisionTreeClassifier",
    "sklearn.ensemble.RandomForestClassifier",
    "numpy.dtype",
]

MODEL_CANDIDATES = [
    Path("./Model/drug_pipeline.skops"),
    Path("./model/drug_pipeline.skops"),
    Path("./Model/drug_pipeline.joblib"),
    Path("./model/drug_pipeline.joblib"),
]

pipe = None
for model_path in MODEL_CANDIDATES:
    if model_path.exists():
        if model_path.suffix == ".skops":
            pipe = sio.load(model_path, trusted=trusted_types)
        else:
            pipe = joblib_load(model_path)
        break


def predict_drug(age, sex, blood_pressure, cholesterol, na_to_k_ratio):
    """Predict drugs based on patient features."""
    if pipe is None:
        return "Model is not available yet. Please try again shortly."

    features_df = pd.DataFrame(
        [
            {
                "Age": age,
                "Sex": sex,
                "BP": blood_pressure,
                "Cholesterol": cholesterol,
                "Na_to_K": na_to_k_ratio,
            }
        ]
    )
    predicted_drug = pipe.predict(features_df)[0]
    return f"Predicted Drug: {predicted_drug}"


inputs = [
    gr.Slider(15, 74, step=1, label="Age"),
    gr.Radio(["M", "F"], label="Sex"),
    gr.Radio(["HIGH", "LOW", "NORMAL"], label="Blood Pressure"),
    gr.Radio(["HIGH", "NORMAL"], label="Cholesterol"),
    gr.Slider(6.2, 38.2, step=0.1, label="Na_to_K"),
]
outputs = [gr.Textbox(label="output")]

examples = [
    [30, "M", "HIGH", "NORMAL", 15.4],
    [35, "F", "LOW", "NORMAL", 8],
    [50, "M", "HIGH", "HIGH", 34],
]


title = "Drug Classification"
description = "Enter the details to correctly identify Drug type?"
article = (
    "This app is a part of the **[Beginner's Guide to CI/CD for Machine Learning]"
    "(https://www.datacamp.com/tutorial/ci-cd-for-machine-learning)**. "
    "It teaches how to automate training, evaluation, and deployment of models "
    "to Hugging Face using GitHub Actions."
)


demo = gr.Interface(
    fn=predict_drug,
    inputs=inputs,
    outputs=outputs,
    examples=examples,
    cache_examples=False,
    title=title,
    description=description,
    article=article,
    theme=gr.themes.Soft(),
)

demo.launch(ssr_mode=False)
