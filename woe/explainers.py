import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Locals
from .utils import annotate_group

from preprocessing import params


class WoEVisualisation:
    def __init__(
        self,
        pred_y,
        h_entailed,
        h_contrast,
        base_lods,
        total_woe,
        attwoes,
        attrib_names,
        class_names,
    ):
        self.prnt_colors = {"pos": "green", "neu": "grey_50", "neg": "red"}
        self.plot_colors = {"pos": "#3C8ABE", "neu": "#808080", "neg": "#CF5246"}
        self.base_lods = base_lods
        self.attrib_names = attrib_names
        self.pred_y = pred_y
        self.attwoes = attwoes
        self.total_woe = total_woe
        self.h_entailed = h_entailed
        self.h_contrast = h_contrast
        self.class_names = class_names
        self.sorted_attwoes = [
            (i, x)
            for x, i in sorted(
                zip(attwoes, range(len(self.attrib_names))), reverse=True
            )
        ]
        self.neg_woes = []
        self.neu_woes = []
        self.pos_woes = []
        for i in range(len(attwoes)):
            if attwoes[i] < 0:
                self.neg_woes.append((i, attwoes[i]))
            elif attwoes[i] == 0:
                self.neu_woes.append((i, attwoes[i]))
            else:
                self.pos_woes.append((i, attwoes[i]))

        self.neg_woes = sorted(self.neg_woes, key=lambda k: k[0], reverse=True)
        self.pos_woes = sorted(self.pos_woes, key=lambda k: k[0], reverse=True)

        self.woe_thresholds = params.WOE_THRESHOLDS

        self.pos_colors = {"pos": "#3C8ABE", "neu": "#808080", "neg": "#CF5246"}

    def plot_bayes(self, figsize=(8, 4), ax=None, show=True, save_path=None):
        if not ax:
            fig, ax = plt.subplots(figsize=figsize)

        lods = [self.base_lods, self.total_woe, 0, self.total_woe + self.base_lods]
        cats = ["PRIOR LOG-ODDS", "TOTAL WOE", "", "POST. LOG-ODDS"]
        ypos = range(len(cats))

        cmap = sns.color_palette("RdBu_r", len(self.woe_thresholds) * 2)[
            ::-1
        ]  # len(vals))

        v = np.fromiter(self.woe_thresholds.values(), dtype="float")
        vals = np.sort(np.concatenate([-v, v, np.array([0])]))
        vals = vals[(vals < 10) & (vals > -10)]

        bar_colors = [cmap[i] for i in np.digitize(lods, vals)]
        ax.barh(ypos, lods, align="center", color=bar_colors)
        ax.set_yticks(ypos)
        ax.set_yticklabels(cats)
        ax.invert_yaxis()  # labels read top-to-bottom

        ax.axvline(0, alpha=1, color="k", linestyle="-")
        ax.set_xlim(
            -1 + min(np.min(self.attwoes), -6), max(6, np.max(self.attwoes)) + 1
        )
        # After Individual WoEs
        ax.axhline(2, alpha=0.5, color="black", linestyle="-")
        ax.set_title("Bayes Posterior Log-Odds Decomposition")

    def plot(
        self,
        figsize=(8, 4),
        ax=None,
        show=True,
        include_lods=False,
        save_path=None,
        mode="neg",
    ):
        # mode: neg, pos, neu, all
        if not ax:
            fig, ax = plt.subplots(figsize=figsize)

        attrib_ord = []
        woe = []
        if mode == "neg":
            if self.neg_woes:
                attrib_ord, woe = list(zip(*self.neg_woes))
        elif mode == "pos":
            if self.pos_woes:
                attrib_ord, woe = list(zip(*self.pos_woes))
        elif mode == "neu":
            if self.neu_woes:
                attrib_ord, woe = list(zip(*self.pos_woes))
        else:
            # attrib_ord, woe = list(zip(*self.sorted_attwoes)) # sorted by woe
            # sorted by feature index
            attrib_ord = range(len(self.attrib_names))
            woe = self.attwoes

        vals = list(woe)  # + [0, self.total_woe]
        cats = [self.attrib_names[i] for i in attrib_ord]  # + ['', 'TOTAL WOE']

        if include_lods:
            vals += [self.base_lods, 0, self.total_woe + self.base_lods]
            cats += ["PRIOR LOG-ODDS", "", "POST. LOG-ODDS"]

        vals = np.array(vals)
        ypos = np.arange(len(cats))
        entailed_str = ",".join([str(self.class_names[c]) for c in self.h_entailed])
        if len(self.h_contrast) == len(self.class_names) - len(self.h_entailed):
            contrast_str = ""
        else:
            contrast_str = (
                "(vs " + ",".join([self.class_names[c] for c in self.h_contrast]) + ")"
            )

        v = np.fromiter(self.woe_thresholds.values(), dtype="float")
        thresholds = np.sort(np.concatenate([-v, v, np.array([0])]))
        thresholds = thresholds[(thresholds < 10) & (thresholds > -10)]

        cmap = sns.color_palette("RdBu_r", len(self.woe_thresholds) * 2)[
            ::-1
        ]  # len(thresholds))

        bar_colors = [cmap[i] for i in np.digitize(vals, thresholds)]
        ax.barh(ypos, vals, align="center", color=bar_colors, zorder=2, height=0.5)

        ax.set_yticks(ypos)
        ax.set_yticklabels(cats, fontsize=17)
        ax.invert_yaxis()  # labels read top-to-bottom
        # ax.set_xlabel('Weight-of-Evidence')
        ax.set_title(
            f"Weight of Evidence for: $\\bf{{{entailed_str}}}$ {contrast_str}",
            fontsize=18,
        )
        ax.set_xlim(
            -1 + min(np.min(self.attwoes), -6), max(6, np.max(self.attwoes)) + 1
        )

        # Horizontal Separators
        ax.axhline(
            len(self.attrib_names), alpha=0.5, color="black", linestyle="-"
        )  # After Individual WoEs
        if include_lods:
            # After Prior LODS
            ax.axhline(
                len(ypos) - 1.5, alpha=0.5, color="black", linestyle=":"
            )  # After LogOdds

        annotate_group(
            "Individual WoE Scores",
            (-0.5, len(self.attrib_names) - 0.5),
            ax,
            orient="v",
            rot=90,
        )

        # Draw helper vertical lines delimiting woe regimes
        ax.axvline(0, alpha=1, color="k", linestyle="-")
        prev_τ = 0
        pad = -0.2
        for i, (level, τ) in enumerate(self.woe_thresholds.items()):
            if -τ > ax.get_xlim()[0]:
                # ax.axvline(-τ, alpha = 0.5, color = 'k', linestyle = '--', zorder=1)
                # -> colored bars
                ax.axvline(
                    -τ,
                    alpha=1,
                    color=cmap[int(len(cmap) / 2) - i - 1],
                    linestyle="--",
                    zorder=1,
                )
            if τ < ax.get_xlim()[1]:
                # ax.axvline(τ, alpha = 0.5, color = 'k', linestyle = '--',  zorder=1) #--> black bars
                # -> colored bars
                ax.axvline(
                    τ,
                    alpha=1,
                    color=cmap[i + int(len(cmap) / 2)],
                    linestyle="--",
                    zorder=1,
                )

            if level == "Neutral":
                annotate_group("Not\nSignificant.", (-τ, τ), ax, pad=pad)
            else:
                annotate_group(
                    "{}\nIn Favor".format(level),
                    (prev_τ, min(τ, ax.get_xlim()[1])),
                    ax,
                    pad=pad,
                )
                annotate_group(
                    "{}\nAgainst".format(level),
                    (max(-τ, ax.get_xlim()[0]), -prev_τ),
                    ax,
                    pad=pad,
                )

            # annotate_group('Significant\nAgainst', (ax.get_xlim()[0],-τ), ax)
            prev_τ = τ
        # ax.text(-2.5,0,'significative against',horizontalalignment='right')
        if save_path:
            plt.savefig(
                save_path + "_expl.pdf", bbox_inches="tight", format="pdf", dpi=100
            )
        if show:
            plt.show()

        return ax


class WoEExplainer:
    """
    Multi-step Contrastive Explainer.
    - alpha, p are parameters for the explanation scoring function - see their description there
    """

    def __init__(
        self,
        woe_model,
        classes=None,
        features=None,
        total_woe_correction=False,
        featgroup_idxs=None,
        featgroup_names=None,
        input_size=None,
        loss_type="norm_delta",
        reg_type="decay",
        alpha=1,
        p=1,
        plot_type="bar",
    ):
        self.classes = classes
        self.features = features
        self.plot_type = plot_type
        self.loss_type = loss_type
        self.reg_type = reg_type
        self.alpha = alpha
        self.p = p
        self.input_size = input_size

        self.woe_model = woe_model
        self.total_woe_correction = total_woe_correction
        self.featgroup_idxs = featgroup_idxs
        self.featgroup_names = featgroup_names
        self.woe_thresholds = params.WOE_THRESHOLDS

    def _get_explanation(self, x, y_pred, hyp, null_hyp, units="groups"):
        """Base function to compute explanation - generic, can take simple or complex hyps
        but a single example at a time.
        """
        assert x.ndim == 1
        assert type(hyp) in [np.ndarray, list]
        assert type(null_hyp) in [np.ndarray, list]

        # Total WOE and Prior Odds
        prior_lodds = self.woe_model.prior_lodds(hyp, null_hyp)

        if "group" in units:
            # Compute Per-Attribute WoE Scores
            woes = []
            for i, idxs in enumerate(self.featgroup_idxs):
                woes.append(self.woe_model.woe(x, hyp, null_hyp, subset=idxs).item())
            woes = np.array(woes).T
            woe_names = self.featgroup_names
            # sorted_attrwoes= [(i,x) for x,i in sorted(zip(attrwoes[idx],range(len(self.attribute_names))), reverse=True)]
        elif units == "features":
            # Compute also per-feature woe for plot
            woes = []
            for i in range(x.shape[0]):
                woes.append(self.woe_model.woe(x, hyp, null_hyp, subset=[i]).item())
            woes = np.array(woes).T
            woe_names = self.features

        # Three methods to get total woe:
        # 1. Direct (will match sum of individual woes if )
        # total_woe   = self.woe_model.woe(x, hyp, null_hyp).item()
        # 2. Sum of individual woes
        total_woe = woes.sum()
        # 3. Prior/Posterior Estimation
        # total_woe = self.woe_model._model_woe(x, hyp, null_hyp).item()

        # Correction on WoE Estimation
        if self.total_woe_correction:
            empirical_woe = self.woe_model._model_woe(x, hyp, null_hyp).item()
            delta = woes.sum() - empirical_woe
            tol = 1e-8
            if delta > tol:
                sum_woes_pos = woes[woes > 0].sum()
                if sum_woes_pos != 0:
                    # woes[woes > 0] -= delta / ((woes > 0).sum())  ## Additive Dampening
                    # Multiplicative Dampening
                    woes[woes > 0] *= (sum_woes_pos - delta) / sum_woes_pos
            elif delta < -tol:
                sum_woes_neg = woes[woes < 0].sum()
                if sum_woes_neg != 0:
                    # woes[woes < 0] -= delta / ((woes < 0).sum())  ## Additive Dampening
                    # Multiplicative Dampening
                    woes[woes < 0] *= (sum_woes_neg - delta) / sum_woes_neg
            # Recompute Total Woe
            total_woe = woes.sum()

        # Create Explanation
        # hyp_classes  = [self.classes[c] for c in hyp]
        # null_classes = [self.classes[c] for c in null_hyp]
        expl = WoEVisualisation(
            y_pred, hyp, null_hyp, prior_lodds, total_woe, woes, woe_names, self.classes
        )

        return expl

    def explain_for_human(
        self,
        x,
        y_pred,
        sequential=False,
        show_ranges=False,
        show_bayes=True,
        units="attributes",
        favor_class="predicted",
        plot=True,
        save_path=None,
    ):
        # If not sequential, this is the only y in numerator. If sequential,
        # this is the class that must be contained in all denominators
        if favor_class is not None and type(favor_class) is int:
            y_num = favor_class
        elif favor_class == "predicted":
            y_num = int(y_pred)
        # elif favor_class == 'true':
        #    y_num = int(y)
        else:
            y_num = 0

        def make_fig(show_ranges, show_bayes):
            ncol, nrow, heights = 1, 1, None
            figsize = (11, 12)
            if show_ranges:
                ncol += 1
            if show_bayes:
                nrow += 1
                heights = [4, 1]

            fig = plt.figure(figsize=figsize)
            gs = gridspec.GridSpec(nrow, ncol, height_ratios=heights)
            # set the spacing between axes.
            gs.update(wspace=0.025, hspace=0.3)

            ax0 = fig.add_subplot(gs[0, 0])
            ax1 = fig.add_subplot(gs[1, 0]) if show_bayes else None
            ax2 = fig.add_subplot(gs[0, 1]) if show_ranges else None
            axes = {"woe": ax0, "bayes": ax1, "range": ax2}

            return fig, axes

        # TODO: Generalize to multiclass
        if not sequential:

            y_num = [y_num]
            y_den = sorted(list(set(range(len(self.classes))) - set(y_num)))

            # Create Explanation
            expl = self._get_explanation(
                x, y_pred, hyp=y_num, null_hyp=y_den, units=units
            )

            if plot:
                # Plot all
                fig, axes = make_fig(show_ranges, show_bayes)
                _ = expl.plot(ax=axes["woe"], show=False, mode="all")
                # if show_ranges:
                #    self.plot_ranges(x, expl, featexpl, range_group, ax = axes['ranges'])
                if show_bayes:
                    expl.plot_bayes(ax=axes["bayes"])

                if save_path:
                    plt.savefig(save_path, bbox_inches="tight", dpi=100)
                plt.close()

        return expl
