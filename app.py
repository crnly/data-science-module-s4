"""Explore the notebook's saved model comparisons with Gradio."""

import json
from pathlib import Path

import gradio as gr
import pandas as pd
from matplotlib.figure import Figure

REPORT = json.loads(Path(__file__).with_name("results.json").read_text())
RESULTS = pd.DataFrame(REPORT["comparison"])


def compare_model(name):
    """Present existing results; model fitting stays in the notebook."""
    rows = RESULTS.loc[RESULTS["Model"] == name]
    best_stage = rows.iloc[0]["Stage"]  # Exported rows retain the notebook's selection order.
    rows = rows.set_index("Stage").loc[REPORT["stages"]]

    figure = Figure(figsize=(8, 3.7), facecolor="#FFFFFF", layout="constrained")
    axis = figure.subplots()
    scores = rows[["Train F1", "Validation F1"]].rename(
        columns={"Train F1": "Training", "Validation F1": "Validation"}
    )
    scores.plot.bar(ax=axis, color=["#FABA8A", "#D35F0A"], width=0.65, rot=0)
    axis.set(facecolor="#FFFFFF", ylim=(0, 1.08), xlabel="", ylabel=REPORT["metric"])
    axis.tick_params(colors="#535152")
    axis.yaxis.label.set_color("#535152")
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.grid(axis="y", color="#ECEBEB", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2,
                frameon=False, labelcolor="#311602")
    for bars in axis.containers:
        axis.bar_label(bars, fmt="%.3f", padding=4, color="#311602", fontsize=9)

    selection_change = rows.loc["Feature selection", "Validation F1"] - rows.loc["Baseline", "Validation F1"]
    tuning_change = rows.loc["Tuned", "Validation F1"] - rows.loc["Feature selection", "Validation F1"]
    gap = rows["F1 gap"].max()
    explanation = (
        f"### Reading this comparison\n"
        f"- **Feature selection:** validation F1 changes by **{selection_change:+.3f}** from baseline.\n"
        f"- **Tuning after selection:** validation F1 changes by **{tuning_change:+.3f}**.\n"
        f"- **Largest training–validation gap:** **{gap:+.3f}**. A larger positive gap can indicate overfitting.\n\n"
        f"Best validation stage within this family: **{best_stage}**. "
        "These are observations on one validation split; small differences do not establish superiority."
    )
    warnings = REPORT["search_warnings"][name]
    if warnings:
        explanation += f"\n\n**Tuning diagnostic:** {warnings} convergence warnings; some fits reached the iteration limit."
    return figure, explanation


def build_demo():
    initial = REPORT["selected"]["model"]
    figure, explanation = compare_model(initial)
    with gr.Blocks(title="Model Comparison", analytics_enabled=False) as demo:
        gr.Markdown(
            '<p class="eyebrow">DATA SCIENCE / MODELING & VALIDATION</p>\n\n'
            '# Model Comparison\n'
            'Explore how feature selection and tuning change the semester models.',
            elem_id="app-header",
        )
        model = gr.Dropdown(REPORT["models"], value=initial, label="Model family", interactive=True)
        plot = gr.Plot(value=figure, label="Training and validation weighted F1", show_label=False)
        reading = gr.Markdown(explanation)
        with gr.Accordion("How the three stages work", open=False):
            gr.Markdown(
                "**Baseline:** fit on all 47 prepared features.\n\n"
                "**Feature selection:** fit on features ranked by a training-only random forest.\n\n"
                "**Tuned:** choose parameters using five grouped training folds, with selection inside each fold.\n\n"
                "Each candidate is then compared on the same supplied validation file. "
                "Weighted F1 combines the two class-specific F1 scores by class frequency."
            )
        gr.Markdown(
            f"Training: {REPORT['rows']['Training']} rows · Validation: {REPORT['rows']['Validation']} rows · "
            "Saved notebook results · Prepared course data limits interpretation.",
            elem_id="app-footnote",
        )
        model.change(compare_model, inputs=model, outputs=[plot, reading],
                     trigger_mode="always_last", api_name="compare_model")
    return demo


# D2 theme 101 (Orange Creamsicle), with the same light palette in either browser mode.
PALETTE = {
    "body_background_fill": "#FFFBF7", "body_text_color": "#311602",
    "body_text_color_subdued": "#535152", "background_fill_primary": "#FFFFFF",
    "background_fill_secondary": "#FFF4EA", "block_background_fill": "#FFFFFF",
    "block_border_color": "#E8D6C8", "block_label_background_fill": "transparent",
    "block_label_text_color": "#535152", "block_title_text_color": "#311602",
    "input_background_fill": "#FFFFFF", "input_background_fill_hover": "#FFF4EA",
    "input_border_color": "#E8D6C8", "input_border_color_focus": "#D35F0A",
}
THEME = gr.themes.Base(
    primary_hue="orange", secondary_hue="orange", neutral_hue="stone",
    font=["Arial", "Helvetica", "sans-serif"],
    font_mono=["ui-monospace", "SFMono-Regular", "Consolas", "monospace"],
    spacing_size="sm",
).set(
    **PALETTE, **{f"{key}_dark": value for key, value in PALETTE.items()},
    color_accent="#D35F0A", color_accent_soft="#FFE0C7", color_accent_soft_dark="#FFE0C7",
    block_radius="2px", input_radius="2px", block_shadow="none", block_shadow_dark="none",
    block_label_text_size="12px", block_label_text_weight="400", input_text_size="14px",
    input_shadow="none", input_shadow_dark="none", input_border_width="1px",
    input_border_width_dark="1px", input_shadow_focus="none", input_shadow_focus_dark="none",
)
CSS = """
.gradio-container { max-width: 940px !important; margin: auto; padding-top: 28px !important; }
#app-header { border-bottom: 1px solid #E8D6C8; padding-bottom: 18px; margin-bottom: 8px; }
#app-header .eyebrow { font: 11px ui-monospace, monospace; letter-spacing: .14em; color: #9B4207; }
#app-header h1 { font-size: 30px !important; font-weight: 400 !important; letter-spacing: -.035em; }
#app-header p { color: #535152; }
.gradio-container input { font-family: ui-monospace, monospace !important; }
#app-footnote { border-top: 1px solid #E8D6C8; margin-top: 8px; padding-top: 12px; }
#app-footnote p { font-size: 12px; color: #535152; }
"""

if __name__ == "__main__":
    build_demo().launch(server_name="127.0.0.1", share=False, theme=THEME, css=CSS, footer_links=[])
