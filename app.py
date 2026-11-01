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
READY = "Ready"

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
        gr.Markdown(
            '<p class="eyebrow">SPORTS WATCH / MODEL EXPLORER</p>\n\n'
            '# Sports Watch Classifier\n'
            'Select a customer, adjust the inputs and run a prediction.',
            elem_id="app-header",
        )
        with gr.Row():
            example = gr.Dropdown(list(EXAMPLES), value=first_name, label="Example customer", interactive=True, scale=3)
            reset = gr.Button("Reset example", scale=1, elem_classes="reset-example")
        with gr.Row(elem_classes="prediction-layout"):
            with gr.Column(scale=3):
                for number, (group, fields) in enumerate(GROUPS.items(), start=1):
                    with gr.Accordion(f"{number:02d} / {group}", open=group != "Watch and service"):
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
                gr.Markdown("### Predicted usage", elem_classes="section-label")
                result = gr.Label(value=READY, label="Predicted model usage", show_label=False, elem_classes="prediction-result")
                submit = gr.Button("Run prediction", variant="primary")
                gr.Markdown("Classifies current watch usage. It does not forecast purchases.")
                gr.Markdown("Blank values use training-based defaults. **Reset example** restores the original inputs.")
        gr.Markdown("Course demonstration · Original example data · Shared IDs and unclear feature timing limit interpretation.", elem_id="app-footnote")
        inputs = [components[column] for column in COLUMNS]
        submit.click(predict, inputs=inputs, outputs=result, api_name="predict")
        for trigger in [example.change, reset.click]:
            trigger(load_example, inputs=example, outputs=inputs + [result], api_visibility="private", queue=False)
        for component in inputs:
            component.input(clear_prediction, outputs=result, api_visibility="private", queue=False)
    return demo


# A restrained black, white and silver palette, shared by light and dark modes.
PALETTE = {
    "body_background_fill": "#090909",
    "body_text_color": "#ededed",
    "body_text_color_subdued": "#a0a0a0",
    "background_fill_primary": "#0d0d0d",
    "background_fill_secondary": "#151515",
    "block_background_fill": "#0d0d0d",
    "block_border_color": "#303030",
    "block_label_background_fill": "transparent",
    "block_label_border_color": "transparent",
    "block_label_text_color": "#a8a8a8",
    "block_title_background_fill": "transparent",
    "block_title_text_color": "#dedede",
    "input_background_fill": "#111111",
    "input_background_fill_hover": "#161616",
    "input_background_fill_focus": "#111111",
    "input_border_color": "#373737",
    "input_border_color_hover": "#696969",
    "input_border_color_focus": "#c2cbd1",
    "input_placeholder_color": "#808080",
    "panel_background_fill": "#0d0d0d",
    "panel_border_color": "#373737",
    "button_primary_background_fill": "#ededed",
    "button_primary_background_fill_hover": "#ffffff",
    "button_primary_border_color": "#ededed",
    "button_primary_border_color_hover": "#ffffff",
    "button_primary_text_color": "#090909",
    "button_primary_text_color_hover": "#090909",
    "button_secondary_background_fill": "#090909",
    "button_secondary_background_fill_hover": "#1b1b1b",
    "button_secondary_border_color": "#484848",
    "button_secondary_border_color_hover": "#a0a0a0",
    "button_secondary_text_color": "#dedede",
    "button_secondary_text_color_hover": "#ffffff",
}
THEME = gr.themes.Base(
    primary_hue="gray", secondary_hue="gray", neutral_hue="gray",
    font=["Arial", "Helvetica", "sans-serif"],
    font_mono=["ui-monospace", "SFMono-Regular", "Consolas", "monospace"],
    spacing_size="sm",
).set(
    **PALETTE, **{f"{name}_dark": value for name, value in PALETTE.items()},
    color_accent="#c2cbd1", color_accent_soft="#202326", color_accent_soft_dark="#202326",
    block_radius="2px", container_radius="2px", input_radius="2px",
    block_label_radius="0px", block_label_border_width="0px", block_label_border_width_dark="0px",
    block_label_shadow="none", block_label_text_size="12px", block_label_text_weight="400",
    block_shadow="none", block_shadow_dark="none", block_padding="14px",
    block_title_text_weight="400", section_header_text_weight="400",
    input_shadow="none", input_shadow_dark="none", input_text_size="14px",
    input_border_width="1px", input_border_width_dark="1px",
    button_border_width="1px", button_border_width_dark="1px",
    input_shadow_focus="none", input_shadow_focus_dark="none",
    button_large_radius="2px", button_medium_radius="2px", button_small_radius="2px",
    button_large_text_size="14px", button_large_text_weight="500",
    button_primary_shadow="none", button_primary_shadow_dark="none",
    button_secondary_shadow="none", button_secondary_shadow_dark="none",
    body_text_size="14px", prose_text_size="14px", layout_gap="16px",
)
CSS = """
.gradio-container { max-width: 1120px !important; margin: auto; padding-top: 30px !important; }
#app-header { border-bottom: 1px solid #303030; padding-bottom: 22px; margin-bottom: 8px; }
#app-header .eyebrow { font: 11px ui-monospace, monospace; letter-spacing: .14em; color: #909090; margin-bottom: 16px; }
#app-header h1 { font-size: 30px !important; font-weight: 400 !important; letter-spacing: -.035em; }
#app-header p { color: #a0a0a0; }
.reset-example { align-self: flex-end !important; margin-bottom: 14px !important; }
.section-label h3 { font: 11px ui-monospace, monospace !important; text-transform: uppercase; letter-spacing: .1em; color: #a0a0a0; }
.prediction-result { border: 0 !important; background: transparent !important; padding: 10px 0 22px !important; }
.prediction-result h2 { justify-content: flex-start !important; padding: 0 !important; text-align: left !important; font-size: 25px !important; font-weight: 400 !important; letter-spacing: -.03em; }
.gradio-container .form { border: 0 !important; background: transparent !important; }
.gradio-container input, .gradio-container textarea { font-family: ui-monospace, SFMono-Regular, Consolas, monospace !important; }
#app-footnote { border-top: 1px solid #303030; margin-top: 8px; padding-top: 16px; }
#app-footnote p { font-size: 12px; color: #909090; }
@media (min-width: 850px) {
    .prediction-panel { position: sticky; top: 20px; align-self: flex-start; }
}
@media (max-width: 849px) {
    .prediction-layout { flex-direction: column-reverse !important; }
}
"""

if __name__ == "__main__":
    build_demo().launch(
        server_name="127.0.0.1", share=False,
        theme=THEME, css=CSS, footer_links=[],
    )
