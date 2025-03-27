"""Weight of Evidence (WoE) visualization and explanation utilities.

This module provides classes for visualizing and explaining Weight of Evidence
results, including plotting functions and explanation generation.
"""

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Optional, Union, Tuple, Any

# Local imports
from .woe_utils import annotate_group
from preprocessing import params
from ice.explainer import Explainer


class WoEVisualisation:
    """Class for visualizing Weight of Evidence results.

    This class provides methods for plotting WoE values, including bar charts
    and Bayesian decomposition visualizations.
    """

    def __init__(
        self,
        pred_y: int,
        h_entailed: List[int],
        h_contrast: List[int],
        base_lods: float,
        total_woe: float,
        attwoes: np.ndarray,
        attrib_names: List[str],
        class_names: List[str],
        woe_order: str = "index",
        show_significant: bool = False,
    ) -> None:
        """Initialize the WoE visualization.

        Args:
            pred_y: Predicted class index
            h_entailed: Entailed hypothesis indices
            h_contrast: Contrast hypothesis indices
            base_lods: Base log odds
            total_woe: Total Weight of Evidence
            attwoes: WoE values for each attribute
            attrib_names: Attribute names
            class_names: Class names
            woe_order: Sort the woe based on ("index" or "value")
        """
        # Define color schemes
        self.prnt_colors = {"pos": "green", "neu": "grey_50", "neg": "red"}
        self.plot_colors = {"pos": "#3C8ABE", "neu": "#808080", "neg": "#CF5246"}

        # Store input parameters
        self.base_lods = base_lods
        self.attrib_names = attrib_names
        self.pred_y = pred_y
        self.attwoes = attwoes
        self.woe_indices = list(range(len(self.attwoes)))
        self.total_woe = total_woe
        self.h_entailed = h_entailed
        self.h_contrast = h_contrast
        self.class_names = class_names
        self.woe_order = woe_order
        self.woe_thresholds = params.WOE_THRESHOLDS

        if show_significant:
            filtered_woes = []
            filtered_woe_indices = []
            filtered_woe_names = []
            for i, val in enumerate(self.attwoes):
                if abs(val) > self.woe_thresholds["Neutral"]:
                    filtered_woes.append(val)
                    filtered_woe_indices.append(i)
                    filtered_woe_names.append(self.attrib_names[i])
            self.attrib_names = filtered_woe_names
            self.attwoes = filtered_woes
            self.woe_indices = filtered_woe_indices

        # Sort WoE values for visualization
        self.sorted_attwoes = [
            (i, x)
            for x, i in sorted(
                zip(attwoes, range(len(self.attrib_names))), reverse=True
            )
        ]

        # Categorize WoE values by sign
        self._categorize_woes()

        # Create color map based on WoE thresholds
        self.cmap = sns.color_palette("RdBu_r", len(params.WOE_THRESHOLDS) * 2)[::-1]

    def _set_color_bars(self, values_to_bin):
        v = np.fromiter(self.woe_thresholds.values(), dtype="float")
        thresholds = np.sort(np.concatenate([-v, v, np.array([0])]))
        thresholds = thresholds[(thresholds < 10) & (thresholds > -10)]
        bar_colors = [self.cmap[i] for i in np.digitize(values_to_bin, thresholds)]
        return bar_colors

    def _categorize_woes(self) -> None:
        """Categorize WoE values into positive, negative, and neutral groups."""
        self.neg_woes = []
        self.neu_woes = []
        self.pos_woes = []

        for i, val in enumerate(self.attwoes):
            if val < 0:
                self.neg_woes.append((self.woe_indices[i], val))
            elif val == 0:
                self.neu_woes.append((self.woe_indices[i], val))
            else:
                self.pos_woes.append((self.woe_indices[i], val))

        if self.woe_order == "index":
            key_index = 0
        elif self.woe_order == "value":
            key_index = 1

        self.neg_woes = sorted(self.neg_woes, key=lambda k: k[key_index], reverse=True)
        self.pos_woes = sorted(self.pos_woes, key=lambda k: k[key_index], reverse=True)

    def plot_bayes(
        self,
        figsize: Tuple[int, int] = (8, 4),
        ax: Optional[plt.Axes] = None,
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> plt.Axes:
        """Plot Bayesian decomposition of log odds.

        Args:
            figsize: Figure size (width, height)
            ax: Matplotlib axes to plot on (created if None)
            show: Whether to display the plot
            save_path: Path to save the figure

        Returns:
            Matplotlib axes with the plot
        """
        if not ax:
            fig, ax = plt.subplots(figsize=figsize)

        # Define data to plot
        lods = [self.base_lods, self.total_woe, 0, self.total_woe + self.base_lods]
        cats = ["PRIOR LOG-ODDS", "TOTAL WOE", "", "POST. LOG-ODDS"]
        ypos = range(len(cats))

        # Color bars based on their value
        bar_colors = self._set_color_bars(values_to_bin=lods)
        ax.barh(ypos, lods, align="center", color=bar_colors)

        # Set axis labels and ticks
        ax.set_yticks(ypos)
        ax.set_yticklabels(cats)
        ax.invert_yaxis()  # labels read top-to-bottom

        # Add reference line and set axis limits
        ax.axvline(0, alpha=1, color="k", linestyle="-")
        ax.set_xlim(
            -1 + min(np.min(self.attwoes), -6), max(6, np.max(self.attwoes)) + 1
        )

        # Add horizontal separator and title
        ax.axhline(2, alpha=0.5, color="black", linestyle="-")
        ax.set_title("Bayes Posterior Log-Odds Decomposition")

        return ax

    def _shorter_annotation(self, level: str, evidence_type: str) -> str:
        """Generate shorter annotation for significance levels.

        Args:
            level: Significance level (e.g., "Decisive", "Strong")
            evidence_type: Direction of evidence ("positive" or "negative")

        Returns:
            Short annotation string
        """
        if evidence_type == "negative":
            annotation = "-"
        elif evidence_type == "positive":
            annotation = "+"

        if level == "Decisive":
            annotation = annotation * 3
        elif level == "Strong":
            annotation = annotation * 2
        elif level == "Substantial":
            annotation = annotation * 1

        return annotation

    def plot(
        self,
        figsize: Tuple[int, int] = (8, 4),
        ax: Optional[plt.Axes] = None,
        show: bool = True,
        include_lods: bool = False,
        save_path: Optional[str] = None,
        attrib_ord: List = [],
        woe: List = [],
    ) -> plt.Axes:
        """Plot Weight of Evidence values as a horizontal bar chart.

        Args:
            figsize: Figure size (width, height)
            ax: Matplotlib axes to plot on (created if None)
            show: Whether to display the plot
            include_lods: Whether to include log odds in the plot
            save_path: Path to save the figure
            evidence_type: Evidence type ("negative", "positive" or "all")

        Returns:
            Matplotlib axes with the plot
        """
        if not ax:
            fig, ax = plt.subplots(figsize=figsize)

        # Prepare data for plotting
        vals = list(woe)
        cats = [self.attrib_names[i] for i in attrib_ord]

        # Add log odds information if requested
        if include_lods:
            vals += [self.base_lods, 0, self.total_woe + self.base_lods]
            cats += ["PRIOR LOG-ODDS", "", "POST. LOG-ODDS"]

        vals = np.array(vals)
        ypos = np.arange(len(cats))

        # Format class names for title
        entailed_str = ",".join([str(self.class_names[c]) for c in self.h_entailed])
        if len(self.h_contrast) == len(self.class_names) - len(self.h_entailed):
            contrast_str = ""
        else:
            contrast_str = (
                "(vs " + ",".join([self.class_names[c] for c in self.h_contrast]) + ")"
            )

        # Plot bars with colors based on value
        bar_colors = self._set_color_bars(values_to_bin=vals)
        ax.barh(ypos, vals, align="center", color=bar_colors, zorder=2, height=0.5)

        # Set axis labels and ticks
        ax.set_yticks(ypos)
        ax.set_yticklabels(cats, fontsize=17)
        ax.invert_yaxis()  # labels read top-to-bottom

        # Set title and limits
        ax.set_title(
            f"Weight of Evidence for: $\\bf{{{entailed_str}}}$ {contrast_str}",
            fontsize=18,
        )
        ax.set_xlim(
            -1 + min(np.min(self.attwoes), -6), max(6, np.max(self.attwoes)) + 1
        )

        # Add horizontal separators
        ax.axhline(len(self.attrib_names), alpha=0.5, color="black", linestyle="-")
        if include_lods:
            ax.axhline(len(ypos) - 1.5, alpha=0.5, color="black", linestyle=":")

        # Add group annotation
        annotate_group(
            "Individual WoE Scores",
            (-0.5, len(self.attrib_names) - 0.5),
            ax,
            orient="v",
            rot=90,
        )

        # Draw vertical threshold lines
        self._draw_threshold_lines(ax)

        return ax

    def _draw_threshold_lines(
        self,
        ax: plt.Axes,
        pad: float = -0.2,
        shift: float = 0.5,
        rot: float = 0,
        evidence_type: str = "all",
        text_size: int = 30,
    ) -> None:
        """Draw vertical lines to mark WoE thresholds.

        Args:
            ax: Matplotlib axes to draw on
        """
        # Draw zero line
        ax.axvline(0, alpha=1, color="k", linestyle="-")

        # Draw threshold lines
        prev_threshold = 0

        for i, (level, threshold) in enumerate(self.woe_thresholds.items()):
            # Draw negative threshold line if within limits
            if -threshold > ax.get_xlim()[0]:
                ax.axvline(
                    -threshold,
                    alpha=1,
                    color=self.cmap[int(len(self.cmap) / 2) - i - 1],
                    linestyle="--",
                    zorder=1,
                )

            # Draw positive threshold line if within limits
            if threshold < ax.get_xlim()[1]:
                ax.axvline(
                    threshold,
                    alpha=1,
                    color=self.cmap[i + int(len(self.cmap) / 2)],
                    linestyle="--",
                    zorder=1,
                )

            # Add annotations
            if level == "Neutral":
                annotate_group(
                    "N",
                    (-threshold, threshold),
                    ax,
                    pad=pad,
                    shift=shift,
                    text_size=text_size,
                )
            else:
                if evidence_type == "all":
                    pos_annotation = self._shorter_annotation(level, "positive")
                    neg_annotation = self._shorter_annotation(level, "negative")

                    # Add positive annotation if within limits
                    annotate_group(
                        f"{pos_annotation}",
                        span=(prev_threshold, min(threshold, ax.get_xlim()[1])),
                        ax=ax,
                        pad=pad,
                        shift=shift,
                        rot=rot,
                        text_size=text_size,
                    )

                    # Add negative annotation if within limits
                    annotate_group(
                        f"{neg_annotation}",
                        span=(max(-threshold, ax.get_xlim()[0]), -prev_threshold),
                        ax=ax,
                        pad=pad,
                        shift=shift,
                        rot=rot,
                        text_size=text_size,
                    )
                else:  # either positive or negative
                    if evidence_type == "positive":
                        span = (prev_threshold, min(threshold, ax.get_xlim()[1]))
                        text_color = "blue"
                    else:
                        span = (max(-threshold, ax.get_xlim()[0]), -prev_threshold)
                        text_color = "red"
                    annotation = self._shorter_annotation(level, evidence_type)
                    annotation = "{}\n({})".format(level, annotation)
                    annotate_group(
                        f"{annotation}",
                        color=text_color,
                        span=span,
                        ax=ax,
                        pad=pad,
                        shift=shift,
                        rot=rot,
                        text_size=text_size,
                    )

            prev_threshold = threshold

    def plot_for_images(
        self,
        original_x: Optional[Any],
        original_h: Optional[Any],
        Exp: Optional[Explainer],
        axsLeft: plt.Axes,
        axsRight: plt.Axes,
        evidence_type: str = "negative",
        min_xlim: int = -9,
        max_xlim: int = 9,
        woe_values: List = [],
        woe_indices: List = [],
        concept_algo: str = "ice",
        img_evidence_path: str = None,
        feature_path: str = None,
    ):
        num_selected_features = rows = len(woe_values)
        if num_selected_features == 0:
            return plt.figure()  # empty figure

        bar_colors = self._set_color_bars(values_to_bin=woe_values)
        rot = 45
        if evidence_type == "positive":
            axsLeft.set_xlim([0, max_xlim])
        elif evidence_type == "negative":
            axsLeft.set_xlim([min_xlim, 0])
        else:
            rot = 0
            axsLeft.set_xlim([min_xlim, max_xlim])

        axsLeft.barh(
            np.arange(num_selected_features),
            woe_values,
            align="center",
            color=bar_colors,
        )

        if concept_algo == "ice":
            axsLeft.set_yticks(
                np.arange(num_selected_features),
                labels=[
                    "{}".format(params.SKIN_FEATURE_ID_TO_LABEL[f]) for f in woe_indices
                ],
            )
        else:
            axsLeft.set_yticks(
                np.arange(num_selected_features),
                labels=[params.SKIN_PCBM_CONCEPT_NAMES[f] for f in woe_indices],
            )
        axsLeft.set_xticks([])
        axsLeft.set_xticklabels([])

        self._draw_threshold_lines(
            ax=axsLeft,
            pad=0,
            shift=0,
            rot=rot,
            evidence_type=evidence_type,
            text_size=13,
        )

        if axsRight.ndim == 1:
            axsRight = axsRight.reshape(1, -1)
        for i, feat in enumerate(woe_indices[::-1]):
            img_test_feat = img_evidence_path / ("feature_{}.jpg".format(feat))
            if not os.path.exists(img_test_feat):
                img_test_feat = Exp.segment_concept_image(
                    original_x, original_h, feat, img_test_feat
                )
            img_test = plt.imread(img_test_feat)
            axsRight[i][0].imshow(img_test, interpolation="none")
            axsRight[i][0].get_xaxis().set_ticks([])
            axsRight[i][0].get_yaxis().set_ticks([])

            img_feat = feature_path / ("{}.jpg".format(feat))
            img = plt.imread(img_feat)
            axsRight[i][1].imshow(img, interpolation="none")
            axsRight[i][1].get_xaxis().set_ticks([])
            axsRight[i][1].get_yaxis().set_ticks([])

        axsRight[rows - 1][0].set_xlabel("Test image")
        axsRight[rows - 1][1].set_xlabel("Training images")
        return axsLeft, axsRight


class WoEExplainer:
    """Multi-step Contrastive Explainer for Weight of Evidence.

    This class provides methods for generating and visualizing Weight of Evidence
    explanations for classification results.
    """

    def __init__(
        self,
        woe_model: Any,
        classes: List[str],
        features: List[str],
        total_woe_correction: bool = False,
        featgroup_idxs: Optional[List[List[int]]] = None,
        featgroup_names: Optional[List[str]] = None,
    ) -> None:
        """Initialize the WoE explainer.

        Args:
            woe_model: Weight of Evidence model
            classes: Class names
            features: Feature names
            total_woe_correction: Whether to correct total WoE
            featgroup_idxs: Feature group indices
            featgroup_names: Feature group names
        """
        self.classes = classes
        self.features = features
        self.woe_model = woe_model
        self.total_woe_correction = total_woe_correction
        self.featgroup_idxs = featgroup_idxs
        self.featgroup_names = featgroup_names
        self.woe_thresholds = params.WOE_THRESHOLDS

    def _get_explanation(
        self,
        x: np.ndarray,
        hypothesis: int,
        hyp: Union[List[int], np.ndarray],
        null_hyp: Union[List[int], np.ndarray],
        units: str = "features",
        show_significant: bool = False,
    ) -> WoEVisualisation:
        """Compute explanation for a single example.

        Args:
            x: Input features
            hypothesis: Hypothesis class
            hyp: Hypothesis indices
            null_hyp: Null hypothesis indices
            units: Units for explanation ("group" or "features")

        Returns:
            WoEVisualisation object
        """
        assert x.ndim == 1, "Input must be a single example (1D array)"
        assert isinstance(hyp, (np.ndarray, list)), "Hypothesis must be array or list"
        assert isinstance(
            null_hyp, (np.ndarray, list)
        ), "Null hypothesis must be array or list"

        num_features = x.shape[0]

        # Calculate prior log odds
        prior_lodds = self.woe_model.prior_lodds(hyp, null_hyp)

        # Compute WoE scores based on units
        if "group" in units:
            # Compute per-attribute WoE scores
            woes = []
            for i, idxs in enumerate(self.featgroup_idxs):
                S = idxs
                T = np.array([feats for feats in self.featgroup_idxs if feats != idxs])
                woes.append(self.woe_model.woe(x, hyp, null_hyp, S=S, T=T))
            woes = np.array(woes).T
            woe_names = self.featgroup_names

        elif units == "features":
            # Compute per-feature WoE scores
            woes = []
            for i in range(num_features):
                S = np.array([i])
                T = np.array(list(set(range(num_features)) - {i}))
                woes.append(self.woe_model.woe(x, hyp, null_hyp, S=S, T=T))
            woes = np.array(woes).T
            woe_names = self.features

        # Calculate total WoE (sum of individual WoEs)
        total_woe = woes.sum()

        # Apply correction if needed
        if self.total_woe_correction:
            self._apply_woe_correction(woes, x, hyp, null_hyp)
            total_woe = woes.sum()

        # Create and return explanation
        return WoEVisualisation(
            pred_y=hypothesis,
            h_entailed=hyp,
            h_contrast=null_hyp,
            base_lods=prior_lodds,
            total_woe=total_woe,
            attwoes=woes,
            attrib_names=woe_names,
            class_names=self.classes,
            show_significant=show_significant,
        )

    def _apply_woe_correction(
        self,
        woes: np.ndarray,
        x: np.ndarray,
        hyp: Union[List[int], np.ndarray],
        null_hyp: Union[List[int], np.ndarray],
    ) -> None:
        """Apply correction to ensure WoE values sum to empirical WoE.

        Args:
            woes: WoE values to correct (modified in-place)
            x: Input features
            hyp: Hypothesis indices
            null_hyp: Null hypothesis indices
        """
        empirical_woe = self.woe_model._model_woe(x, hyp, null_hyp)
        delta = woes.sum() - empirical_woe
        tol = 1e-8

        if delta > tol:
            # Positive discrepancy - adjust positive WoEs
            sum_woes_pos = woes[woes > 0].sum()
            if sum_woes_pos != 0:
                # Multiplicative dampening
                woes[woes > 0] *= (sum_woes_pos - delta) / sum_woes_pos
        elif delta < -tol:
            # Negative discrepancy - adjust negative WoEs
            sum_woes_neg = woes[woes < 0].sum()
            if sum_woes_neg != 0:
                # Multiplicative dampening
                woes[woes < 0] *= (sum_woes_neg - delta) / sum_woes_neg

    def explain_for_human(
        self,
        x: np.ndarray,
        hypothesis: int,
        show_ranges: bool = False,
        show_bayes: bool = False,
        units: str = "features",
        plot: bool = True,
        save_path: Optional[str] = None,
        data_type: str = "tabular",
        evidence_type: str = "all",
        show_significant: bool = False,
        img_evidence_path: Optional[str] = None,
        feature_path: Optional[str] = None,
        original_x: Optional[Any] = None,
        original_h: Optional[Any] = None,
        Exp: Optional[Explainer] = None,
    ) -> WoEVisualisation:
        """Generate human-friendly explanation for a prediction.

        Args:
            x: Input features
            hypothesis: Hypothesis class
            show_ranges: Whether to show ranges in the plot
            show_bayes: Whether to show Bayesian decomposition
            units: Units for explanation ("attributes" or "features")
            plot: Whether to generate plots
            save_path: Path to save plots
            data_type: Input data type ("tabular" or "image")
            evidence_type: ("positive" or "negative" or "all")
            img_evidence_path: Path to save evidence images
            feature_path: Path to save feature images
            show_significant: Whether to show significant evidence not in "Neutral" range
            original_x: Original input features
            original_h: Original mask image
            Exp: Explainer object

        Returns:
            WoEVisualisation object
        """
        # Generate explanation
        y_num = [hypothesis]
        y_den = sorted(list(set(range(len(self.classes))) - set(y_num)))

        # Create explanation
        expl = self._get_explanation(
            x,
            hypothesis,
            hyp=y_num,
            null_hyp=y_den,
            units=units,
            show_significant=show_significant,
        )

        # Generate plots if requested
        if plot:
            if data_type == "tabular":
                self._generate_plots(
                    expl=expl,
                    show_ranges=show_ranges,
                    show_bayes=show_bayes,
                    save_path=save_path,
                    data_type=data_type,
                    evidence_type=evidence_type,
                )
            elif data_type == "image":
                self._generate_plots(
                    expl=expl,
                    show_ranges=show_ranges,
                    show_bayes=show_bayes,
                    save_path=save_path,
                    data_type=data_type,
                    evidence_type=evidence_type,
                    img_evidence_path=img_evidence_path,
                    feature_path=feature_path,
                    original_x=original_x,
                    original_h=original_h,
                    Exp=Exp,
                )

        return expl

    def _generate_plots(
        self,
        expl: WoEVisualisation,
        show_ranges: bool,
        show_bayes: bool,
        save_path: Optional[str] = None,
        data_type: str = "tabular",
        evidence_type: str = "all",
        img_evidence_path: Optional[str] = None,
        feature_path: Optional[str] = None,
        original_x: Optional[Any] = None,
        original_h: Optional[Any] = None,
        Exp: Optional[Explainer] = None,
    ) -> None:
        """Generate and display plots for the explanation.

        Args:
            expl: WoEVisualisation object
            show_ranges: Whether to show ranges in the plot
            show_bayes: Whether to show Bayesian decomposition
            save_path: Path to save plots
            data_type: tabular or image
        """
        # Select attributes based on evidence_type
        attrib_ord = []
        woe = []
        if evidence_type == "negative":
            if expl.neg_woes:
                attrib_ord, woe = list(zip(*expl.neg_woes))
        elif evidence_type == "positive":
            if expl.pos_woes:
                attrib_ord, woe = list(zip(*expl.pos_woes))
        else:  # evidence_type == "all"
            attrib_ord = expl.woe_indices
            woe = expl.attwoes

        if data_type == "tabular":
            # Setup figure and axes
            ncol, nrow, heights = 1, 1, None
            figsize = (11, 12)

            if show_ranges:
                ncol += 1
            if show_bayes:
                nrow += 1
                heights = [4, 1]

            fig = plt.figure(figsize=figsize)
            gs = gridspec.GridSpec(nrow, ncol, height_ratios=heights)
            gs.update(wspace=0.025, hspace=0.3)

            # Create axes
            ax0 = fig.add_subplot(gs[0, 0])
            ax1 = fig.add_subplot(gs[1, 0]) if show_bayes else None
            ax2 = fig.add_subplot(gs[0, 1]) if show_ranges else None

            # Generate plots
            expl.plot(ax=ax0, show=False, attrib_ord=attrib_ord, woe=woe)
            if show_bayes:
                expl.plot_bayes(ax=ax1)
        elif data_type == "image":
            if len(woe) == 0:
                print("No evidence")
                return
            fig = plt.figure(layout="constrained", figsize=(16, len(woe) * 2))
            subfigs = fig.subfigures(1, 2, wspace=0.01, width_ratios=[1, 2])
            axsLeft = subfigs[0].subplots(1, 1)
            axsRight = subfigs[1].subplots(
                nrows=len(woe), ncols=2, sharey=True, width_ratios=[1, 5]
            )

            expl.plot_for_images(
                axsLeft=axsLeft,
                axsRight=axsRight,
                evidence_type=evidence_type,
                woe_values=woe,
                woe_indices=attrib_ord,
                img_evidence_path=img_evidence_path,
                feature_path=feature_path,
                original_x=original_x,
                original_h=original_h,
                Exp=Exp,
            )
        # Save or show figure
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=100)
        else:
            plt.show()

        plt.close()
