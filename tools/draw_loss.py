#!/usr/bin/env python3

import argparse
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def main():
    parser = argparse.ArgumentParser(
        description="Plot training metrics from CSV."
    )
    parser.add_argument(
        "csv_file",
        help="Path to CSV file"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="training_metrics.html",
        help="Output HTML file"
    )
    parser.add_argument(
        "--logy",
        action="store_true",
        help="Use log scale for loss/MAE axis"
    )
    args = parser.parse_args()

    # Load CSV
    df = pd.read_csv(args.csv_file)

    # Use row index as training step
    df["step"] = range(len(df))

    mae_cols = [
        "training energy MAE (per atom)",
        "training forces MAE",
        "training virial MAE (per atom)",
        "training non_conservative_forces MAE (per atom)",
        "training non_conservative_stress MAE",
    ]

    fig = make_subplots(
        rows=1,
        cols=1,
        specs=[[{"secondary_y": True}]],
    )

    # Training loss
    fig.add_trace(
        go.Scatter(
            x=df["step"],
            y=df["training loss"],
            mode="lines",
            name="training loss",
            customdata=df["Epoch"],
            hovertemplate=(
                "Step=%{x}<br>"
                "Epoch=%{customdata}<br>"
                "training loss=%{y:.6g}"
                "<extra></extra>"
            ),
        ),
        secondary_y=False,
    )

    # Learning rate
    fig.add_trace(
        go.Scatter(
            x=df["step"],
            y=df["learning rate"],
            mode="lines",
            name="learning rate",
            line=dict(dash="dash"),
            customdata=df["Epoch"],
            hovertemplate=(
                "Step=%{x}<br>"
                "Epoch=%{customdata}<br>"
                "learning rate=%{y:.6g}"
                "<extra></extra>"
            ),
        ),
        secondary_y=True,
    )

    # MAE metrics
    for col in mae_cols:
        if col not in df.columns:
            continue

        fig.add_trace(
            go.Scatter(
                x=df["step"],
                y=df[col],
                mode="lines",
                name=col,
                customdata=df["Epoch"],
                hovertemplate=(
                    "Step=%{x}<br>"
                    "Epoch=%{customdata}<br>"
                    f"{col}=%{{y:.6g}}"
                    "<extra></extra>"
                ),
            ),
            secondary_y=False,
        )

    fig.update_layout(
        title="Training Metrics",
        template="plotly_white",
        hovermode="x unified",
        height=900,
        width=1600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    fig.update_xaxes(
        title_text="Training Step",
        showgrid=True,
    )

    fig.update_yaxes(
        title_text="Loss / MAE",
        secondary_y=False,
        type="log" if args.logy else "linear",
    )

    fig.update_yaxes(
        title_text="Learning Rate",
        secondary_y=True,
    )

    # Save interactive HTML
    fig.write_html(
        args.output,
        include_plotlyjs=True,
    )

    print(f"Saved plot to: {args.output}")


if __name__ == "__main__":
    main()