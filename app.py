"""Run the saved course model with a small Gradio interface."""

from pathlib import Path

import cloudpickle
import gradio as gr
import numpy as np
import pandas as pd

with Path(__file__).with_name("model.pkl").open("rb") as model_file:
    bundle = cloudpickle.load(model_file)

COLUMNS = bundle["columns"]
NUMERIC = bundle["numeric_columns"]
EXAMPLES = dict(zip([f"Customer {identifier}" for identifier in bundle["example_ids"]], bundle["examples"]))
READY = "Ready to predict"

GROUPS = {
    "Customer": [
        ("Age of customer", "Age"),
        ("Sex", "Sex"),
        ("Ctry", "Country"),
        ("Town", "Town"),
        ("Customer Years", "Customer years"),
    ],
    "Training": [
        ("Swimming Hours per Week", "Swimming hours / week"),
        ("Biking Hours per Week", "Biking hours / week"),
        ("Running Hours per Week", "Running hours / week"),
        ("Total Training Hours per Week", "Total training hours / week"),
        ("VO2 Max", "VO₂ max"),
        ("10k Running Time Prediction", "Predicted 10k running time"),
        ("Calories Burned per Week", "Calories burned / week"),
        ("Goal of Training", "Training goal"),
        ("Preferred Training Daytime", "Preferred training time"),
    ],
    "Watch and service": [
        ("Most current software update", "Latest software update (YYYY-MM-DD)"),
        ("Support Cases of Customer", "Support cases"),
        ("Subscription Type", "Subscription"),
        ("Color of Watch", "Watch color"),
        ("Synchronisation", "Synchronisation"),
    ],
}


def predict(*values):
    """Parse form values; all feature engineering stays in the saved pipeline."""
    frame = pd.DataFrame([values], columns=COLUMNS)
    if frame.replace("", np.nan).isna().all(axis=None):
        raise gr.Error("Enter a customer value or select one of the examples.")
    try:
        frame[NUMERIC] = frame[NUMERIC].astype(float)
    except (TypeError, ValueError):
        raise gr.Error("Use numbers in numeric fields, or leave them blank.") from None
    if np.isinf(frame[NUMERIC].to_numpy()).any():
        raise gr.Error("Use finite numbers in numeric fields, or leave them blank.")
    categorical = [column for column in COLUMNS if column not in NUMERIC]
    frame[categorical] = frame[categorical].astype(object).replace("", np.nan)
    frame[categorical] = frame[categorical].where(frame[categorical].notna(), np.nan)
    label = int(bundle["pipeline"].predict(frame)[0])
    return {0: "Older model", 1: "Latest model"}[label]


def load_example(name):
    row = EXAMPLES[name]
    return [row[column] if pd.notna(row[column]) else None for column in COLUMNS] + [READY]


def clear_prediction():
    return READY


def build_demo():
    first_name = next(iter(EXAMPLES))
    first_row = EXAMPLES[first_name]
    components = {}
    with gr.Blocks(title="Sports Watch Classifier", analytics_enabled=False) as demo:
        gr.Markdown("# Sports Watch Classifier\nExplore a course model of current watch usage. Select an example, edit the values and predict.")
        with gr.Row():
            example = gr.Dropdown(list(EXAMPLES), value=first_name, label="Original example customer", interactive=True, scale=3)
            reset = gr.Button("Reset example", scale=1)
        with gr.Row(elem_classes="prediction-layout"):
            with gr.Column(scale=3):
                for group, fields in GROUPS.items():
                    with gr.Accordion(group, open=group != "Watch and service"):
                        if group == "Training":
                            gr.Markdown("Total hours should match the three sports. Inconsistent totals are treated as missing.")
                        for start in range(0, len(fields), 2):
                            with gr.Row():
                                for column, label in fields[start:start + 2]:
                                    value = first_row[column]
                                    if column in NUMERIC:
                                        component = gr.Number(value=value, label=label)
                                    elif column in bundle["categories"]:
                                        component = gr.Dropdown(bundle["categories"][column], value=value, label=label, allow_custom_value=True)
                                    else:
                                        component = gr.Textbox(value=value, label=label, placeholder="YYYY-MM-DD")
                                    components[column] = component
            with gr.Column(scale=1, min_width=270, variant="panel", elem_classes="prediction-panel"):
                gr.Markdown("### Prediction")
                result = gr.Label(value=READY, label="Predicted model usage")
                submit = gr.Button("Predict watch usage", variant="primary")
                gr.Markdown("Classifies use of an **older** or **latest** watch model. It does not predict future purchases.")
                gr.Markdown("Missing values use training-based imputation. You can reset any edits with **Reset example**.")
        gr.Markdown("*Course demonstration. Examples retain the original data values. Shared IDs across the course files and unclear feature timing limit the model’s interpretation.*")
        inputs = [components[column] for column in COLUMNS]
        submit.click(predict, inputs=inputs, outputs=result, api_name="predict")
        for trigger in [example.change, reset.click]:
            trigger(load_example, inputs=example, outputs=inputs + [result], api_visibility="private", queue=False)
        for component in inputs:
            component.input(clear_prediction, outputs=result, api_visibility="private", queue=False)
    return demo


if __name__ == "__main__":
    build_demo().launch(
        server_name="127.0.0.1", share=False,
        theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate", font=["system-ui", "sans-serif"]),
        css="""
        .gradio-container { max-width: 1120px !important; margin: auto; }
        @media (min-width: 850px) {
            .prediction-panel { position: sticky; top: 20px; align-self: flex-start; }
        }
        @media (max-width: 849px) {
            .prediction-layout { flex-direction: column-reverse !important; }
        }
        """,
        footer_links=[],
    )
