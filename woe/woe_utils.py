import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available")


def int2hot(idx, n=None):
    """Convert integer indices to one-hot encoded tensor.

    Args:
        idx: Integer or tensor of integers
        n: Number of classes (optional, defaults to max index + 1)

    Returns:
        One-hot encoded tensor
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for this function")

    if type(idx) is int:
        idx = torch.tensor([idx])
    if idx.dim() == 0:
        idx = idx.view(-1, 1)
    if not n:
        n = idx.max() + 1
    return torch.zeros(len(idx), n).scatter_(1, idx.unsqueeze(1), 1.0)


def annotate_group(
    name,
    span,
    ax=None,
    orient="h",
    pad=None,
    shift=0.5,
    rot=0,
    color="black",
    text_size=30,
):
    """Annotates a span of the x-axis (or y-axis if orient ='v')

    Args:
        name: Text to display
        span: Tuple of (start, end) values
        ax: Matplotlib axis (optional)
        orient: 'h' for horizontal or 'v' for vertical
        pad: Padding for annotation (optional)
        shift: Shift from axis (optional)
        rot: Rotation angle (optional)
        color: Text color (optional)

    Returns:
        Tuple of (left_arrow, right_arrow) annotation objects
    """

    def annotate(ax, name, left, right, y, pad):
        xy = (left, y) if orient == "h" else (y, left)
        xytext = (right, y - pad) if orient == "h" else (y + pad, right)
        valign = "top" if orient == "h" else "center"
        halign = "center" if orient == "h" else "center"

        if orient == "h":
            connectionstyle = "angle,angleB=90,angleA=0,rad=5"
        else:
            connectionstyle = "angle,angleB=0,angleA=-90,rad=5"

        arrow = ax.annotate(
            name,
            color=color,
            xy=xy,
            xycoords="data",
            xytext=xytext,
            textcoords="data",
            annotation_clip=False,
            verticalalignment=valign,
            horizontalalignment=halign,
            linespacing=2.0,
            arrowprops=dict(
                arrowstyle="-", shrinkA=0, shrinkB=0, connectionstyle=connectionstyle
            ),
            fontsize=text_size,
            fontweight="bold",
            rotation=rot,
        )
        return arrow

    if ax is None:
        ax = plt.gca()
    lim = ax.get_ylim()[0] if orient == "h" else ax.get_xlim()[1]
    min = lim + (shift if orient == "h" else shift)
    center = np.mean(span)

    if pad is None:
        pad = 0.01 if orient == "h" else 0.2
    left_arrow = annotate(ax, name, span[0], center, min, pad)
    right_arrow = annotate(ax, name, span[1], center, min, pad)
    return left_arrow, right_arrow


def range_plot(
    X,
    x0=None,
    colnames=None,
    plottype="box",
    groups=None,
    x0_labels=None,
    color_values=True,
    ax=None,
):
    """Create a range plot comparing feature values.

    Args:
        X: Feature matrix
        x0: Reference point to plot (optional)
        colnames: Column names for features
        plottype: 'box' or 'swarm' plot type
        groups: Group labels for coloring (optional)
        x0_labels: Labels for reference point (optional)
        color_values: Whether to color values based on thresholds
        ax: Matplotlib axis (optional)

    Returns:
        Matplotlib axis with the plot
    """
    plot_colors = {"pos": "#3C8ABE", "neu": "#808080", "neg": "#CF5246"}
    palette = sns.color_palette("Set2")

    assert X.shape[1] == len(x0)

    X = X.copy()
    if x0 is not None:
        x0 = x0.copy()

    df = pd.DataFrame(X, columns=colnames)
    pargs = {}

    if groups is not None:
        df["groups"] = groups
        df = df.melt(id_vars=["groups"])
        pargs["hue"] = "groups"
        pargs["x"] = "value"
        pargs["y"] = "variable"

    if not ax:
        fig, ax = plt.subplots(figsize=(8, 8))

    if plottype == "swarm":
        ax = sns.swarmplot(data=df, orient="h", palette="Set2", ax=ax, alpha=0.5)
    else:
        ax = sns.boxplot(
            data=df, orient="h", **pargs, palette=palette, showfliers=False, ax=ax
        )
        ax.set_xlabel(None)
        for patch in ax.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, 0.2))

    if x0 is not None:
        (line,) = ax.plot(
            x0,
            range(X.shape[1]),
            "kd",
            linestyle="dashed",
            ms=5,
            linewidth=0.3,
            zorder=1000,
        )

    if x0_labels is not None:
        xmin, xmax = ax.get_xlim()
        delta = xmax - xmin
        pad = 0.1 * delta
        ax.set_xlim(xmin, xmax + pad)

        for i, val in enumerate(x0_labels):
            if isinstance(val, (float, np.float64, np.float32)):
                if color_values:
                    cstr = "neg" if (val <= -2) else ("pos" if val >= 2 else "neu")
                else:
                    cstr = "neu"
                txt = f"{val:2.2e}"
            else:
                cstr = "neu"
                txt = f"{val:2}"

            ax.text(
                xmax + 0.6 * pad,
                i,
                txt,
                fontsize=10,
                zorder=1001,
                ha="right",
                color=plot_colors[cstr],
            )

    if groups is not None:
        handles, labels = ax.get_legend_handles_labels()
        ncol = 2
        if x0 is not None:
            handles.append(line)
            labels.append("This example")
            ncol += 1
        ax.legend(
            handles=handles,
            labels=labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.025),
            ncol=ncol,
        )

    ax.set_title("Feature values of explained example vs training data")
    return ax
