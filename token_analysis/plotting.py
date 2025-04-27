import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_token_per_word_plotly(ratio_data, plots_dir):
    """
    Create and save a bar plot comparing token-per-word ratios across tokenizers and languages.

    Parameters
    ----------
    ratio_data : list
        List of tuples with tokenizer name, language, and token-per-word ratio
    plots_dir : str
        Directory to save the plot
    """
    df = pd.DataFrame(ratio_data, columns=["Tokenizer", "Language", "Tokens per Word"])

    fig = px.bar(
        df,
        x="Tokenizer",
        y="Tokens per Word",
        color="Language",
        barmode="group",
        text="Tokens per Word",
        title="Token per Word Ratio by Tokenizer and Language",
        height=600,
        width=900
    )

    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(
        yaxis_title="Tokens per Word (lower is better)",
        xaxis_title="Tokenizer",
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        template='plotly_white',
    )

    output_path = os.path.join(plots_dir, "token_per_word_comparison_plotly.png")
    fig.write_image(output_path)
    fig.show()
    print(f"   • Saved token-per-word comparison plot to '{output_path}'")


def plot_token_length_distribution_plotly(token_length_stats, plots_dir):
    """
    Create and save histograms showing token length distribution across languages and tokenizers.

    Parameters
    ----------
    token_length_stats : dict
        Dictionary with token length statistics
    plots_dir : str
        Directory to save the plot
    """
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("English Token Length Distribution", "Hindi Token Length Distribution"))

    # Plot for English
    for name, stats in token_length_stats.items():
        fig.add_trace(
            go.Histogram(
                x=stats["en"]["all_lengths"],
                name=name,
                histnorm='probability density',
                opacity=0.6
            ),
            row=1, col=1
        )

    # Plot for Hindi
    for name, stats in token_length_stats.items():
        fig.add_trace(
            go.Histogram(
                x=stats["hi"]["all_lengths"],
                name=name,
                histnorm='probability density',
                opacity=0.6,
                showlegend=False  # Hide duplicate legends on second plot
            ),
            row=1, col=2
        )

    fig.update_layout(
        title_text="Token Length Distribution (English vs Hindi)",
        height=500,
        width=1000,
        template='plotly_white',
        barmode='overlay'
    )

    fig.update_xaxes(title_text="Number of Tokens", row=1, col=1)
    fig.update_yaxes(title_text="Density", row=1, col=1)
    fig.update_xaxes(title_text="Number of Tokens", row=1, col=2)

    output_path = os.path.join(plots_dir, "token_length_distribution_plotly.png")
    fig.write_image(output_path)
    fig.show()
    print(f"   • Saved token length distribution plot to '{output_path}'")