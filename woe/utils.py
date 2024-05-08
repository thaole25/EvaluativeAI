import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def annotate_group(name, span, ax=None, orient="h", pad=None, shift=0.5, rot=0):
    """Annotates a span of the x-axis (or y-axis if orient ='v')"""

    def annotate(ax, name, left, right, y, pad):
        xy = (left, y) if orient == "h" else (y, left)
        xytext = (right, y - pad) if orient == "h" else (y + pad, right)
        valign = "top" if orient == "h" else "center"
        halign = "center" if orient == "h" else "center"
        # rot = 0 if orient == 'h' else 90
        if orient == "h":
            connectionstyle = "angle,angleB=90,angleA=0,rad=5"
        else:
            connectionstyle = "angle,angleB=0,angleA=-90,rad=5"

        arrow = ax.annotate(
            name,
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
            fontsize=12,
            rotation=rot,
        )
        return arrow

    if ax is None:
        ax = plt.gca()
    lim = ax.get_ylim()[0] if orient == "h" else ax.get_xlim()[1]
    min = lim + (shift if orient == "h" else shift)
    center = np.mean(span)
    # pad = 0.01 * np.ptp(lim) # I had this but seems to be always 0
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
    """
    If provided, groups should be an array of same rowlength of X, will be used as hue
    """
    plot_colors = {"pos": "#3C8ABE", "neu": "#808080", "neg": "#CF5246"}

    # This works only for binary, and assumes 0,1 in groups are neg/pos.
    # palette = sns.color_palette([plot_colors[v] for v in ['neg', 'pos']])

    # palette = sns.color_palette([plot_colors[v] for v in ['neg', 'pos']])
    palette = sns.color_palette("Set2")

    assert X.shape[1] == len(x0)

    X = X.copy()
    if x0 is not None:
        x0 = x0.copy()

    # if rescale:
    #     # We rescale so that boxplots are roughly aligned
    #     # (do so only for nonbinary fetaures)
    #     centers = np.median(X[:,self.nonbinary_feats], axis=0)
    #     X[:,self.nonbinary_feats] -= centers
    #
    #     scales = np.std(X[:,self.nonbinary_feats], axis=0)
    #     #scales = np.quantile(X[:,self.nonbinary_feats], .75, axis=0)
    #     X[:,self.nonbinary_feats[scales > 0]] /= scales[scales>0]
    #
    #     # Must also rescale x0, but display it's tr
    #     x[self.nonbinary_feats]   -= centers
    #     x[self.nonbinary_feats[scales > 0]] /= scales[scales>0]

    df = pd.DataFrame(X, columns=colnames)

    pargs = {}

    if groups is not None:
        # df = pd.concat([df, pd.DataFrame({'groups': groups})])
        df["groups"] = groups
        #
        # df = pd.DataFrame(np.hstack([X, self.Y[:,None].astype(int)]),
        #                   columns = list(self.features[feat_order]) + ['class'])
        # Will need to melt to plot splitting vy var
        # pdb.set_trace()
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
        )  # , boxprops=dict(alpha=.3))
        # ax.xaxis.set_label_text("")
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
        # farright = xmax - 0.5
        ax.set_xlim(xmin, xmax + pad)  # Give some buffer to point labels
        for i, val in enumerate(x0_labels):
            # ax.text(x0[i]+0.5, i, txt, fontsize=10, zorder = 1000)
            if type(val) in [float, np.float64, np.float32]:
                if color_values:
                    cstr = "neg" if (val <= -2) else ("pos" if val >= 2 else "neu")
                else:
                    cstr = "neu"
                txt = "{:2.2e}".format(val)
            else:
                cstr = "neu"
                txt = "{:2}".format(val)
            # txt = '{:2.2f}'.format(val) if type(val) is float else '{}'.format(val)
            # print(x0[i], val)
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
        # Get rid of legend title
        handles, labels = ax.get_legend_handles_labels()
        ncol = 2
        if x0 is not None:
            # Also, add points to legend
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
