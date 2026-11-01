"""Run the saved course model with a small Gradio interface."""

from pathlib import Path

import cloudpickle
import gradio as gr
import numpy as np
import pandas as pd

with Path(__file__).with_name("model.pkl").open("rb") as model_file:
    bundle = cloudpickle.load(model_file)

COLUMNS = bundle["columns"]
EXAMPLES = dict(zip(bundle["example_names"], bundle["examples"]))
READY = "Ready"


def predict(features):
    """Use the same prepared feature schema as the semester CSV files."""
    if features.shape != (len(COLUMNS), 2) or features.iloc[:, 0].tolist() != COLUMNS:
        raise gr.Error("Use all original feature names in their original order. Reset the example.")
    try:
        values = pd.to_numeric(features.iloc[:, 1], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError):
        raise gr.Error("Every prepared feature needs a numeric value.") from None
    if not np.isfinite(values).all():
        raise gr.Error("Every prepared feature needs a finite numeric value.")
    categorical_groups = {}
    for column, value in zip(COLUMNS, values):
        if column.startswith("cat__"):
            if value not in (0, 1):
                raise gr.Error("One-hot category values must be 0 or 1.")
            group = column.rsplit("_", 1)[0]
            categorical_groups[group] = categorical_groups.get(group, 0) + value
    if any(total > 1 for total in categorical_groups.values()):
        raise gr.Error("Use at most one active category in each one-hot group.")
    frame = pd.DataFrame([values], columns=COLUMNS)
    label = int(bundle["pipeline"].predict(frame)[0])
    return {0: "Older model", 1: "Latest model"}[label]


def load_example(name):
    return pd.DataFrame({"Feature": COLUMNS, "Value": [EXAMPLES[name][column] for column in COLUMNS]}), READY


def clear_prediction():
    return READY


def build_demo():
    first_name = next(iter(EXAMPLES))
    with gr.Blocks(title="Sports Watch Classifier", analytics_enabled=False) as demo:
        gr.Markdown(
            '<p class="eyebrow">SPORTS WATCH / MODEL EXPLORER</p>\n\n'
            '# Sports Watch Classifier\n'
            'Explore the semester model with three original prepared test examples.',
            elem_id="app-header",
        )
        with gr.Row():
            example = gr.Dropdown(list(EXAMPLES), value=first_name, label="Prepared example", interactive=True, scale=3)
            reset = gr.Button("Reset example", scale=1, elem_classes="reset-example")
        with gr.Row(elem_classes="prediction-layout"):
            with gr.Column(scale=3):
                gr.Markdown(
                    f"### {bundle['model_name']}\n"
                    f"Stage: **{bundle['stage']}** · Training + validation: **781 rows**\n\n"
                    "Select an example and run the saved model. The notebook contains the full model comparison."
                )
                with gr.Accordion("Prepared features / optional editing", open=False):
                    gr.Markdown(
                        "`num__` values are already transformed, not raw measurements. "
                        "`cat__` values are one-hot categories (0 or 1). "
                        "Edits explore the model response; they do not demonstrate causal effects."
                    )
                    features = gr.Dataframe(
                        value=load_example(first_name)[0], headers=["Feature", "Value"],
                        datatype=["str", "number"], type="pandas", interactive=True,
                        static_columns=[0], row_count=len(COLUMNS), column_count=2,
                        column_widths=["78%", "22%"],
                        max_height=340, label="Model inputs", show_label=False,
                        buttons=[],
                    )
            with gr.Column(scale=1, min_width=270, variant="panel", elem_classes="prediction-panel"):
                gr.Markdown("### Predicted usage", elem_classes="section-label")
                result = gr.Label(value=READY, label="Predicted model usage", show_label=False, elem_classes="prediction-result")
                submit = gr.Button("Run prediction", variant="primary")
                gr.Markdown("Classifies current watch usage. It does not forecast purchases.")
                gr.Markdown("**Reset example** restores all original feature values.")
        gr.Markdown("Course demonstration · Prepared semester data · Earlier preprocessing and person-level independence are not verified.", elem_id="app-footnote")
        submit.click(predict, inputs=features, outputs=result, api_name="predict")
        for trigger in [example.change, reset.click]:
            trigger(load_example, inputs=example, outputs=[features, result], api_visibility="private", queue=False)
        features.input(clear_prediction, outputs=result, api_visibility="private", queue=False)
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
    "table_border_color": "#303030",
    "table_even_background_fill": "#0d0d0d",
    "table_odd_background_fill": "#151515",
    "table_row_focus": "#202326",
    "table_text_color": "#ededed",
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
