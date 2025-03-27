"""Weight of Evidence (WoE) implementation using Gaussian distributions.

This module provides functionality for computing Weight of Evidence using Gaussian
distributions, including both independent and dependent variable cases.
"""

import numpy as np
import torch
from scipy.special import logsumexp
import time
from typing import Optional, Union, List, Tuple, Dict, Any

from preprocessing import params


def gaussian_log_density(
    x: torch.Tensor,
    S: np.ndarray,
    T: np.ndarray,
    hypothesis: int,
    means: torch.Tensor,
    covs: torch.Tensor,
    is_independent: bool,
) -> torch.Tensor:
    """Compute log density of Gaussian distribution.

    Args:
        x: Input data tensor
        S: Selected feature indices
        T: Complementary feature indices
        hypothesis: Class index
        means: Class means
        covs: Class covariances
        is_independent: Whether to use independent Gaussian model

    Returns:
        Log density value
    """
    x_S = x[:, S]
    d = len(S)
    covH = covs[hypothesis, :]
    if covH.shape[0] == 1:
        covH = covH.squeeze(0)
    meanS = means[hypothesis, S]
    meanT = means[hypothesis, T]

    if is_independent:
        cov = torch.diag(covH)[S]
        log_density = -0.5 * torch.log(2 * np.pi * cov) - ((x_S - meanS) ** 2) / (
            2 * cov
        )
    else:
        # Dependent variables - conditional distribution
        covSS = covs[hypothesis, S][:, S]
        covST = covs[hypothesis, S][:, T]
        covTT = covs[hypothesis, T][:, T]
        covTT_inv = torch.linalg.inv(covTT)

        covSTT_inv = covST @ covTT_inv
        conditional_cov = covSS - covSTT_inv @ covST.T
        cov_det = torch.linalg.det(conditional_cov)
        cov_inv = torch.linalg.inv(conditional_cov)

        conditional_mean = meanS + (x[:, T] - meanT) @ covSTT_inv.T
        diff = x_S - conditional_mean
        log_density = (
            -0.5 * (d * torch.log(torch.tensor(2 * np.pi)) + torch.log(cov_det))
            - 0.5 * diff @ cov_inv @ diff.T
        )

    return log_density


def gaussian_mixture(
    x: torch.Tensor,
    S: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    priors: torch.Tensor,
    means: torch.Tensor,
    covs: torch.Tensor,
    is_independent: bool,
) -> float:
    """Compute log probability of Gaussian mixture model.

    Args:
        x: Input data tensor
        S: Selected feature indices
        T: Complementary feature indices
        Y: Class indices
        priors: Class priors
        means: Class means
        covs: Class covariances
        is_independent: Whether to use independent Gaussian model

    Returns:
        Log mixture probability
    """
    log_probs = []
    for k in Y:
        log_priors = torch.log(priors[k])
        log_gaussian = gaussian_log_density(
            means=means,
            covs=covs,
            is_independent=is_independent,
            hypothesis=k,
            x=x,
            S=S,
            T=T,
        )
        log_probs.append((log_priors + log_gaussian).cpu().numpy())

    return logsumexp(log_probs)


class WoEGaussian:
    """Weight of Evidence implementation using Gaussian distributions.

    This class implements Weight of Evidence (WoE) using Gaussian distributions,
    supporting both independent and dependent variable cases.
    """

    def __init__(
        self,
        classifier_model: Any,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        no_features: int,
        woe_clf: str,
        classes: List[str],
        is_independent: bool = False,
    ) -> None:
        """Initialize WoE Gaussian model.

        Args:
            classifier_model: Base classifier model
            X: Training features
            y: Training labels
            no_features: Number of features
            woe_clf: Type of WoE classifier
            classes: Class names
            is_independent: Whether to use independent Gaussian model
        """
        print("Processing WOE Gaussian model...")
        self.model = classifier_model
        self.classes = classes
        self.is_independent = is_independent
        self.X = X
        self.y = y
        self.d = no_features

        if woe_clf == "original":
            self.fit(self.X, self.y)
        else:
            y_preds = self.model.predict(self.X)
            self.fit(self.X, y_preds)

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        Y: Union[np.ndarray, torch.Tensor],
        eps: float = 1e-6,
    ) -> None:
        """Fit the Gaussian model to training data.

        Args:
            X: Training features
            Y: Training labels
            eps: Small constant for numerical stability
        """
        Y = self._process_hypothesis(Y)
        self.priors = []
        self.means = []
        self.covs = []

        for c in params.TARGET_CLASSES:
            s = time.time()
            # Priors
            prior = (Y == c).sum() / len(Y)
            self.priors.append(prior)

            # Class data and compute mean
            X_per_class = X[Y == c]
            X_per_class = torch.tensor(X_per_class).to(device=params.DEVICE)
            X_mean = X_per_class.mean(axis=0)
            self.means.append(X_mean)

            # Compute regularised covariance
            if X_per_class.shape[0] <= 1:
                X_covs = torch.zeros(self.d, self.d).to(device=params.DEVICE)
            else:
                X_covs = torch.cov(X_per_class.T)
            delta = max(eps - torch.linalg.eigvalsh(X_covs).min(), 0)
            X_covs += torch.eye(self.d).to(device=params.DEVICE) * delta
            self.covs.append(X_covs)

            t = time.time()
            print(f"Prediction: {c}, {X_per_class.shape}, {t - s:.2f}s")

        self.priors = torch.tensor(self.priors, device=params.DEVICE)
        self.means = torch.stack(self.means)
        self.covs = torch.stack(self.covs)
        print("Priors: ", self.priors)
        print("Means: ", self.means)
        print("Covs: ", self.covs)

    def _process_hypothesis(
        self, y: Union[int, List[int], np.ndarray, set]
    ) -> np.ndarray:
        """Convert hypothesis to numpy array format.

        Args:
            y: Hypothesis in various formats

        Returns:
            Numpy array of hypothesis indices
        """
        if isinstance(y, int):
            return np.array([y])
        elif isinstance(y, (list, set)):
            return np.array(list(y))
        return y

    def _process_inputs(
        self,
        x: Optional[Union[np.ndarray, torch.Tensor]] = None,
        y1: Optional[Union[int, List[int], np.ndarray]] = None,
        y2: Optional[Union[int, List[int], np.ndarray]] = None,
        subset: Optional[Union[List[int], np.ndarray, set]] = None,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """Process and validate input arguments.

        Args:
            x: Input features
            y1: Primary hypothesis
            y2: Alternative hypothesis
            subset: Feature subset indices

        Returns:
            Tuple of processed inputs
        """
        if y1 is not None:
            y1 = self._process_hypothesis(y1)

        if y2 is None and y1 is not None:
            y2 = np.array(list(set(self.classes).difference(set(y1))))
        elif y2 is not None:
            y2 = self._process_hypothesis(y2)

        if x is not None and x.ndim == 1:
            x = x.reshape(1, -1)

        if subset is not None:
            if isinstance(subset, (list, set)):
                subset = np.array(list(subset))

        return x, y1, y2, subset

    def _convert_type_to_number(
        self, x: Union[torch.Tensor, np.ndarray, List[float]]
    ) -> float:
        """Convert tensor/array to scalar number.

        Args:
            x: Input tensor/array

        Returns:
            Scalar value
        """
        while isinstance(x, (list, np.ndarray)):
            x = x[0]
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        return float(x)

    def woe(
        self,
        x: Union[np.ndarray, torch.Tensor],
        y1: Union[int, List[int], np.ndarray],
        y2: Union[int, List[int], np.ndarray],
        S: np.ndarray,
        T: np.ndarray,
        **kwargs,
    ) -> float:
        """Compute Weight of Evidence.

        Args:
            x: Input features
            y1: Primary hypothesis
            y2: Alternative hypothesis
            S: Selected feature indices
            T: Complementary feature indices
            **kwargs: Additional arguments

        Returns:
            Weight of Evidence value
        """
        x, y1, y2, S = self._process_inputs(x, y1, y2, S)

        ll_num = gaussian_log_density(
            x=x,
            S=S,
            T=T,
            hypothesis=y1,
            means=self.means,
            covs=self.covs,
            is_independent=self.is_independent,
        )

        ll_denom = gaussian_mixture(
            x=x,
            S=S,
            T=T,
            Y=y2,
            priors=self.priors,
            means=self.means,
            covs=self.covs,
            is_independent=self.is_independent,
        )

        ll_num = self._convert_type_to_number(ll_num)
        ll_denom = self._convert_type_to_number(ll_denom)
        return ll_num - ll_denom

    def _model_woe(
        self,
        x: Union[np.ndarray, torch.Tensor],
        y1: Union[int, List[int], np.ndarray],
        y2: Union[int, List[int], np.ndarray],
        **kwargs,
    ) -> float:
        """Compute WoE as difference of posterior and prior log odds.

        Args:
            x: Input features
            y1: Primary hypothesis
            y2: Alternative hypothesis
            **kwargs: Additional arguments

        Returns:
            Weight of Evidence value
        """
        postodds = self.posterior_lodds(x, y1, y2)
        priodds = self.prior_lodds(y1, y2)

        if type(postodds) != type(priodds):
            postodds = self._convert_type_to_number(postodds)
            priodds = self._convert_type_to_number(priodds)

        return postodds - priodds

    def posterior_lodds(
        self,
        x: Union[np.ndarray, torch.Tensor],
        y1: Union[int, List[int], np.ndarray],
        y2: Union[int, List[int], np.ndarray],
        eps: float = 1e-12,
    ) -> float:
        """Compute posterior log odds.

        Args:
            x: Input features
            y1: Primary hypothesis
            y2: Alternative hypothesis
            eps: Small constant for numerical stability

        Returns:
            Posterior log odds value
        """
        y1 = self._process_hypothesis(y1)
        y2 = self._process_hypothesis(y2)
        x = x.cpu().numpy()

        probs = self.model.predict_proba(x.reshape(1, -1))[0]

        odds_num = probs[y1]
        if len(y1) > 1:
            odds_num = odds_num.sum()

        odds_den = probs[y2]
        if len(y2) > 1:
            odds_den = odds_den.sum()

        odds_num = np.clip(odds_num, eps, 1 - eps)
        odds_den = np.clip(odds_den, eps, 1 - eps)

        return np.log(odds_num / odds_den)

    def prior_lodds(
        self,
        y1: Union[int, List[int], np.ndarray],
        y2: Union[int, List[int], np.ndarray],
    ) -> float:
        """Compute prior log odds.

        Args:
            y1: Primary hypothesis
            y2: Alternative hypothesis

        Returns:
            Prior log odds value
        """
        _, y1, y2, _ = self._process_inputs(None, y1, y2)

        odds_num = self.priors[y1]
        if len(y1) > 1:
            odds_num = odds_num.sum()
        else:
            odds_num = odds_num.item()

        odds_den = self.priors[y2]
        if len(y2) > 1:
            odds_den = odds_den.sum()
        else:
            odds_den = odds_den.item()

        if hasattr(odds_num, 'device') and odds_num.device.type == 'cuda':
            odds_num = odds_num.cpu()
        if hasattr(odds_den, 'device') and odds_den.device.type == 'cuda':
            odds_den = odds_den.cpu()

        return np.log(odds_num / odds_den)
