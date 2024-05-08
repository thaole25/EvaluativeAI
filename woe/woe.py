import numpy as np
import scipy as sp
import torch
from scipy.stats import multivariate_normal
from scipy.special import logsumexp

import pdb
import time

from preprocessing import params

DEBUG = False


class WoEWrapper:
    """
    Child methods should specify the following methods:
        - log_prob (computes loglikelhood of X given Y)
        - log_prob_partial (computes loglikelhood of X_S given Y)
    and should instantiate the following attributes:
        - priors (nparray of class prior probabilities)
    """

    def __init__(self, model, task, classes, input_size, complex_hyp_method="mixture"):
        self.model = model
        self.classes = classes
        self.task = task
        self.input_size = input_size
        self.data_transform = None
        self.cache = None
        self.caching = False
        self.priors = None
        self.debug = False
        self.complex_hyp_method = complex_hyp_method

    def _process_hypothesis(self, y):
        if type(y) is int:
            y = np.array([y])
        elif type(y) is list:
            y = np.array(y)
        elif type(y) is set:
            y = np.array(list(y))
        return y

    def _process_inputs(self, x=None, y1=None, y2=None, subset=None):
        """All other functions will assume:
        - x is (n,d) 2-dim array
        - y1, y2 are np.arrays with indices of classes
        """
        if y1 is not None:
            y1 = self._process_hypothesis(y1)

        if y2 is None and y1 is not None:
            # Default to complement as alternate hypothesis
            y2 = np.array(list(set(self.classes).difference(set(y1))))
        elif y2 is not None:
            y2 = self._process_hypothesis(y2)

        if x is not None:
            if x.ndim == 1:
                # Most functions expect more than one sample, so need to reshape.
                # assert x.shape[0] == self.d, (x.shape, self.d)
                # print(x.shape)
                x = x.reshape(1, -1)

        if subset is not None:
            if type(subset) is list:
                subset = np.array(subset)
            elif type(subset) is set:
                subset = np.array(list(subset))

        return x, y1, y2, subset

    def convert_type_to_number(self, x):
        if type(x) is list or type(x) is np.ndarray:
            x = x[0]
        elif type(x) is torch.Tensor:
            x = x.cpu().numpy()
        return x

    def woe(self, x, y1, y2=None, subset=None, **kwargs):
        """
        y1 = selected hypothesis
        y2 = alternative hypothesis
        """
        x, y1, y2, subset = self._process_inputs(x, y1, y2, subset)
        ll_num = self.generalized_conditional_ll(x, y1, subset, **kwargs)
        ll_denom = self.generalized_conditional_ll(x, y2, subset, **kwargs)
        ll_num = self.convert_type_to_number(ll_num)
        ll_denom = self.convert_type_to_number(ll_denom)
        woe = ll_num - ll_denom
        return woe

    def _model_woe(self, x, y1, y2=None, **kwargs):
        """Compute WoE as difference of posterior and prior log odds"""
        postodds = self.posterior_lodds(x, y1, y2)
        priodds = self.prior_lodds(y1, y2)
        if type(postodds) != type(priodds):
            postodds = self.convert_type_to_number(postodds)
            priodds = self.convert_type_to_number(priodds)
        return postodds - priodds

    def prior_lodds(self, y1, y2=None):
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
        return torch.log(odds_num / odds_den)

    def prior(self, ys):
        return self.priors[ys]

    def log_prior(self, ys):
        return torch.log(self.priors[ys])

    def posterior_lodds(self, x, y1, y2=None, eps=1e-12):
        x, y1, y2, _ = self._process_inputs(x, y1, y2)

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

    def generalized_conditional_ll(self, x, y, subset=None, verbose=False, **kwargs):
        """A wrapper to handle different cases for conditional LL computation:
        simple or complex hypothesis.

        Parameters
        ----------
        x : array-like, shape = [n_subset]
            Input
        y : int or list of ints
            If just an int, this is simpl hypthesis ratio, if list it's composite
        Returns
        -------
        log probability : float
        """
        k = len(self.classes)

        # We memoize woe computation for efficiency. For this, need determinsitc hash
        # This has to be done before doing view or reshape on x.
        hash_x = id(x)  # hash(x)  - hash is non determinsitc!
        hash_S = -1 if subset is None else subset[0]  # .item()
        if DEBUG and hash_S != -1:
            print("here", hash_S)

        if (not self.debug) and y.shape[0] == 1:
            if verbose:
                print("H is simple hypotesis")
            if subset is None:
                loglike = self.log_prob(x, y[0])
            else:
                loglike = self.log_prob_partial(x, y[0], subset)
            loglike = (
                loglike.squeeze()
            )  # to get scalar, to be consistent by logsumexp below
        else:
            if verbose:
                print("H is composite hypotesis")

            if self.complex_hyp_method == "average":
                # Averaging Approach - Does satisfy sum desideratum
                if subset is None:
                    loglike = self.log_prob_complex(x, y)
                else:
                    loglike = self.log_prob_complex_partial(x, y, subset)
            elif self.complex_hyp_method == "max":
                # Argmax Approach - Doesnt satisfy sum desideratum
                if subset is None:
                    loglikes = np.array([self.log_prob(x, yi) for yi in y])
                else:
                    loglikes = np.array(
                        [self.log_prob_partial(x, yi, subset) for yi in y]
                    )
                loglike = np.max(loglikes, axis=0)
            elif self.complex_hyp_method == "mixture":
                # Mixture Model Approach - Doesnt satisfy sum desideratum
                # Need to compute log prob with respect to each class in y, then do:
                # log p(x|Y∈C) = log (1/P(Y∈C) * ∑_y p(x|Y=y)p(Y=y) )
                #              = log ∑_y exp{ log p(x|Y=y) + log p(Y=y)} - log P(Y∈C)
                priors = self.prior(np.array(range(k)))
                logpriors = torch.log(priors[y])
                logprior_set = torch.log(priors[y].sum())
                # size npoints x hyp size
                loglike = torch.zeros((x.shape[0], y.shape[0]))
                lognormprob = torch.log(priors[y] / priors[y].sum())

                for i, yi in enumerate(y):
                    if (
                        self.caching
                        and ((hash_x, hash_S) in self.cache)
                        and (yi in self.cache[(hash_x, hash_S)])
                    ):
                        if DEBUG and hash_S != -1:
                            print("using cached: ({},{})".format(hash_x, hash_S))
                        loglike[:, i] = self.cache[(hash_x, hash_S)][yi]
                    else:
                        # yi = int2hot(torch.tensor([yi]*x.shape[0]), k) -> for pytorch
                        if subset is None:
                            # logpriors[i] #- logprior_set
                            loglike[:, i] = self.log_prob(x, yi) + lognormprob[i]
                        else:
                            loglike[:, i] = (
                                self.log_prob_partial(x, yi, subset) + lognormprob[i]
                            )  # + logpriors[i] #- logprior_set
                        if self.caching:
                            if not (hash_x, hash_S) in self.cache:
                                self.cache[(hash_x, hash_S)] = {}

                loglike = logsumexp(loglike, axis=1)  # - logprior_set

        return loglike

    def log_prob(self, x, y):
        """Compute log density of X conditioned on Y.

        Parameters
        ----------
        x : array-like, shape = [n_subset]
            Input
        y : int

        Returns
        -------
        log probability : float
        """
        """
            TODO: use stored log_det_covs, inv_covs for efficiency
        """
        x = self._process_inputs(x)[0]
        assert isinstance(y, (int, np.integer))

        μ = self.means[y, :]
        Σ = self.covs[y, :]

        if self.cond_type == "nb":
            logp = gaussian_logdensity(x, μ, Σ, independent=True)
        else:
            logp = gaussian_logdensity(x, μ, Σ, independent=False)

        return logp

    def log_prob_partial(self, x, y, subset):
        x, _, _, subset = self._process_inputs(x, subset=subset)
        assert isinstance(y, (int, np.integer)), y

        if self.cond_type == "nb":
            # With Naive Bayes assumption, P(X_i | X_j, Y ) = P(X_i | Y),
            # so it suffices to marginalize over subset of features.
            μ = self.means[y, :]
            Σ = self.covs[y, :]
            logp = gaussian_logdensity(x, μ, Σ, marginal=subset, independent=True)

        elif self.cond_type == "full":
            S = subset  # S, the unconditioned variables
            # Complement of S, conditioning variables
            T = list(set(range(self.d)) - set(S))

            # Get Covariance
            # Might want to memoize/cache some of these? Sigma is indep of x.
            # See https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
            # Approach 1: Direct
            Σs = self.covs[y, S][:, S]
            Σst = self.covs[y, S][:, T]  # (|S|,|T|)
            Σt_inv = torch.linalg.inv(self.covs[y, T][:, T])  # (|T|,|T|)

            Σ_stinv = torch.mm(Σst, Σt_inv)  # (|S|,|T|)
            Σ = Σs - torch.mm(Σ_stinv, Σst.T)

            # Approach 2: Uses Schur Complement Identity, but we would still need to compute Σ_stinv to compute mean
            # Σb = np.linalg.inv(self.inv_covs[y,S][:,S])

            assert Σ.shape == (len(S), len(S))
            assert (
                torch.linalg.eig(Σ)[0].cpu().numpy() >= 0
            ).all(), "Marginalized cov not psd"

            # Get Mean
            diff_T = x[:, T] - self.means[y, T]  # (n,|T|) (broadcasting)
            μ = self.means[y, S] - torch.mm(diff_T, Σ_stinv.T)  # (n, |S|)
            # assert μ.shape == (self.n, len(S))

            # =  np.linalg.inv(Σ) by schur complement
            # Σ_inv = self.inv_covs[y, S][:, S]
            # Σ_logdet = np.linalg.slogdet(Σ)[1]

            logp = gaussian_logdensity(x, μ, Σ, independent=False)

        else:
            raise (ValueError)
        return logp

    def covariance_average(self, covs, means=None, weights=None, method="am"):
        if method == "am":
            covs = covs.cpu().numpy()
            # Method 1: Arithmetic Mean
            Σ = np.average(covs, axis=0, weights=weights)
            covs = torch.tensor(covs, device=params.DEVICE)
            Σ = torch.tensor(Σ, device=params.DEVICE)
            # print('Σ Method 1', Σ)
        elif method == "empirical":
            # Σ = np.average(covs, axis = 0, weights=weights)
            # print('Σ Method 1', Σ[:3])
            # Method 2: Recreate empirical covarance matrix
            means = means.cpu().numpy()
            μ = np.average(means, axis=0, weights=weights)
            means = torch.tensor(means, device=params.DEVICE)
            μ = torch.tensor(μ, device=params.DEVICE)
            sum = 0
            for i in range(covs.shape[0]):
                Σi = covs[i, :] if covs[i, :].ndim == 2 else torch.diag(covs[i, :])
                sum += weights[i] * (Σi + torch.mm(means[i, :].T, means[i, :]))
            Σ = sum - torch.mm(μ, μ.T)
            if self.cond_type == "nb":
                assert np.all(Σ.cpu().numpy() > 0), Σ
                Σ = torch.diag(Σ)
            # print('Σ Method 2', Σ[:3])
            # pdb.set_trace()
        else:
            raise ValueError()
        return Σ

    def log_prob_complex(self, x, y):
        x, y = self._process_inputs(x, y)[:2]
        assert len(y) > 1

        weights = self.priors[y] / self.priors[y].sum()
        μ = np.average(self.means[y, :], axis=0, weights=weights)
        Σ = self.covariance_average(
            self.covs[y, :], means=self.means[y, :], weights=weights
        )

        if self.cond_type == "nb":
            logp = gaussian_logdensity(x, μ, Σ, independent=True)
        else:
            logp = gaussian_logdensity(x, μ, Σ, independent=False)

        return logp

    def log_prob_complex_partial(self, x, y, subset):
        x, y, _, subset = self._process_inputs(x, y, subset=subset)
        assert len(y) > 1

        if self.cond_type == "nb":
            weights = self.priors[y] / self.priors[y].sum()
            μ = np.average(self.means[y, :], axis=0, weights=weights)
            Σ = self.covariance_average(
                self.covs[y, :], means=self.means[y, :], weights=weights
            )
            logp = gaussian_logdensity(x, μ, Σ, marginal=subset, independent=True)
        elif self.cond_type == "full":
            raise NotImplementedError()
            # logp = gaussian_logdensity(x, μ, Σ, independent=False)

        return logp


class WoEGaussian(WoEWrapper):
    def __init__(
        self, clf, X, task=None, classes=None, input_size=None, cond_type="full"
    ):
        super().__init__(clf, task, classes, input_size)
        self.cond_type = cond_type  # 'nb' or 'full'
        self.fit(X, clf.predict(X))

    def fit(self, X, Y, eps=1e-8, shift_cov=0.01):
        self.n, self.d = X.shape
        means = []
        covs = []
        inv_covs = []
        log_det_covs = []
        distribs = []
        class_prior = []
        for c in self.classes:
            class_prior.append((Y == c).sum() / len(Y))
            μ = X[Y == c].mean(axis=0)
            means.append(μ)
            Σ = np.cov(X[Y == c].T)
            delta = max(eps - np.linalg.eigvalsh(Σ).min(), 0)
            Σ += np.eye(self.d) * 1.1 * delta  # add to ensure PSD'ness
            # TODO: find more principled smoothing
            Σ += shift_cov * np.eye(self.d)
            try:
                assert np.linalg.eigvalsh(Σ).min() >= eps
            except:
                pdb.set_trace()
            if self.cond_type == "nb":
                # Store digonals only
                Σ = np.diag(Σ)
                Σ_inv = 1 / Σ
                logdet = np.log(Σ).sum()
                try:
                    distribs.append(multivariate_normal(μ, Σ))
                except:
                    pdb.set_trace(header="Failed at creating normal distrib")
            else:
                # Approach 1: Direct
                # Σ_inv = np.linalg.inv(Σ) # Change to pseudoinverse alla scipy
                # s, logdet = np.linalg.slogdet(Σ)
                # assert s >= 0
                # Approach 2: Alla scipy - with (more robust) psuedoinverse
                U, logdet = _psd_pinv_decomposed_log_pdet(Σ, rcond=1e-12)
                # If we use this - maybe move to storting U instead?
                Σ_inv = np.dot(U, U.T)

                try:
                    distribs.append(multivariate_normal(μ, Σ))
                except:
                    pdb.set_trace()

            covs.append(Σ)
            inv_covs.append(Σ_inv)
            log_det_covs.append(logdet)
            assert not np.isnan(Σ).any()

        self.means = np.stack(means)
        self.covs = np.stack(covs)
        self.inv_covs = np.stack(inv_covs)
        self.log_det_covs = np.stack(log_det_covs)
        self.distribs = distribs
        self.priors = np.array(class_prior)

        self.dl2π = self.d * np.log(2.0 * np.pi)


class WoEImageGaussian(WoEWrapper):
    def __init__(
        self,
        method_name,
        classifier_model,
        X,
        y,
        Exp,
        concept_model,
        args,
        layer_name,
        task=None,
        classes=None,
        input_size=None,
        cond_type="full",
    ) -> None:
        super().__init__(classifier_model, task, classes, input_size)
        print("Processing WOE Gaussian model...")
        self.method_name = method_name
        self.cond_type = cond_type  # 'nb' or 'full'
        self.layer_name = layer_name
        self.args = args
        self.X = X
        self.y = y
        # self.n = len(X.dataset)
        self.Exp = Exp
        self.concept_model = concept_model
        self.d = args.no_concepts
        y_preds = self.model.predict(self.X)
        self.fit(self.X, y_preds)

    def fit(self, X, Y, eps=1e-8, shift_cov=0.01):
        Y = self._process_hypothesis(Y)
        self.priors = []
        self.means = []
        self.covs = []
        for c in params.TARGET_CLASSES:
            s = time.time()
            prior = (Y == c).sum() / len(Y)
            self.priors.append(prior)
            X_per_class = X[Y == c]
            X_per_class = torch.tensor(X_per_class).to(device=params.DEVICE)
            X_mean = X_per_class.mean(axis=0)
            self.means.append(X_mean)
            X_covs = torch.cov(X_per_class.T)
            delta = max(eps - torch.linalg.eigvalsh(X_covs).min(), 0)
            X_covs += (
                torch.eye(self.d).to(device=params.DEVICE) * 1.1 * delta
            )  # add to ensure PSD'ness
            # TODO: find more principled smoothing
            X_covs += shift_cov * torch.eye(self.d).to(device=params.DEVICE)
            if self.cond_type == "nb":
                X_covs = torch.diag(X_covs)
            self.covs.append(X_covs)
            t = time.time()
            print("Prediction: ", c, X_per_class.shape, t - s)

        self.priors = torch.tensor(self.priors, device=params.DEVICE)
        self.means = torch.stack(self.means)
        self.covs = torch.stack(self.covs)
        print("Mean shape: ", self.means.shape)
        print("Covs shape: ", self.covs.shape)

    def posterior_lodds(self, x, y1, y2=None, eps=1e-12):
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


# TAKEN FROM SCIPY:
def _pinv_1d(v, eps=1e-5):
    """
    A helper function for computing the pseudoinverse.
    Parameters
    ----------
    v : iterable of numbers
        This may be thought of as a vector of eigenvalues or singular values.
    eps : float
        Elements of v smaller than eps are considered negligible.
    Returns
    -------
    v_pinv : 1d float ndarray
        A vector of pseudo-inverted numbers.
    """
    return torch.tensor([0 if abs(x) < eps else 1 / x for x in v], device=params.DEVICE)


def _psd_pinv_decomposed_log_pdet(
    mat, cond=None, rcond=None, lower=True, check_finite=True
):
    """
    Compute a decomposition of the pseudo-inverse and the logarithm of
    the pseudo-determinant of a symmetric positive semi-definite
    matrix.
    The pseudo-determinant of a matrix is defined as the product of
    the non-zero eigenvalues, and coincides with the usual determinant
    for a full matrix.
    Parameters
    ----------
    mat : array_like
        Input array of shape (`m`, `n`)
    cond, rcond : float or None
        Cutoff for 'small' singular values.
        Eigenvalues smaller than ``rcond*largest_eigenvalue``
        are considered zero.
        If None or -1, suitable machine precision is used.
    lower : bool, optional
        Whether the pertinent array data is taken from the lower or upper
        triangle of `mat`. (Default: lower)
    check_finite : boolean, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    Returns
    -------
    M : array_like
        The pseudo-inverse of the input matrix is np.dot(M, M.T).
    log_pdet : float
        Logarithm of the pseudo-determinant of the matrix.
    """
    # Compute the symmetric eigendecomposition.
    # The input covariance matrix is required to be real symmetric
    # and positive semidefinite which implies that its eigenvalues
    # are all real and non-negative,
    # but clip them anyway to avoid numerical issues.

    # TODO: the code to set cond/rcond is identical to that in
    # scipy.linalg.{pinvh, pinv2} and if/when this function is subsumed
    # into scipy.linalg it should probably be shared between all of
    # these routines.

    # Note that eigh takes care of array conversion, chkfinite,
    # and assertion that the matrix is square.
    s, u = sp.linalg.eigh(mat, lower=lower, check_finite=check_finite)

    if rcond is not None:
        cond = rcond
    if cond in [None, -1]:
        t = u.dtype.char.lower()
        factor = {"f": 1e3, "d": 1e6}
        cond = factor[t] * torch.finfo(t).eps
    eps = cond * torch.max(abs(s))

    if torch.min(s) < -eps:
        raise ValueError("the covariance matrix must be positive semidefinite")

    s_pinv = _pinv_1d(s, eps)
    U = torch.multiply(u, torch.sqrt(s_pinv))
    log_pdet = torch.sum(torch.log(s[s > eps]))

    return U, log_pdet


def gaussian_logdensity(
    x,
    μ,
    Σ,
    Σ_inv=None,
    logdetΣ=None,
    marginal=None,
    independent=False,
):
    """

    - marginal (ndarray or list): if provided will return marginal density of these dimensions
    """
    x_, μ_, Σ_ = x, μ, Σ

    d = μ.shape[0]

    def _extract_diag(M):
        assert M.ndim <= 2
        if M.ndim == 2:
            return torch.diag(M)
        elif M.ndim == 1:
            return M

    if Σ.ndim == 2:  # Used for not independent input
        assert Σ.shape == (d, d)
        _isdiagΣ = torch.count_nonzero(Σ - torch.diag(torch.diagonal(Σ))) == 0
    elif not independent:
        raise ValueError()
    else:
        # Σ was passed as diagonal
        assert Σ.shape == (d,)
        _isdiagΣ = True

    if independent and _isdiagΣ:
        Σ_ = _extract_diag(Σ_)
    elif independent and (not _isdiagΣ):
        print(
            "Warning: independent=True but Σ is not diagonal. Will treat as such anyways."
        )
        Σ_ = _extract_diag(Σ_)
        Σ_inv, logdetΣ = None, None
    elif (not independent) and _isdiagΣ:
        # Maybe user forgot this is independent? Better recompute inv and logdet in this case
        independent = True
        Σ_inv, logdetΣ = None, None

    # Check for PSD-ness of Covariance
    # if independent:
    #     # By this point, independent case should have Σ as vector
    #     assert (Σ_.ndim == 1) and np.all(Σ_ >= 0), Σ_
    # else:
    #     # Checking for psd'ness might be too expensive
    #     assert (Σ_.ndim == 2)

    if marginal is not None:
        # Marginalizing a multivariate gaussian -> dropping dimensions
        Σ_ = Σ_[marginal] if independent else Σ_[marginal, :][:, marginal]
        μ_ = μ_[marginal]
        x_ = x_[:, marginal]
        Σ_inv, logdetΣ = None, None  # Can't use these anymore
        d = len(marginal)

    if logdetΣ is None or Σ_inv is None:
        if independent:
            # Σ_inv =
            logdetΣ = torch.log(Σ_).sum()
        else:
            U, logdetΣ = _psd_pinv_decomposed_log_pdet(Σ_, rcond=1e-12)
            Σ_inv = torch.mm(U, U.T)

    diff = x_ - μ_  # (x - μ_j) , size (n,d)

    if independent:
        expterm = torch.sum((diff**2) / Σ_, 1)
    else:
        expterm = torch.diag(torch.mm(torch.mm(diff, Σ_inv), diff.T))
        # TODO: Faster and cleaner way to do this by storing sqrt of precision matrix. See
        # https://github.com/scipy/scipy/blob/v0.14.0/scipy/stats/_multivariate.py line 330

    constant = torch.tensor(d * np.log(2.0 * np.pi)).to(device=params.DEVICE) + logdetΣ

    logp = -0.5 * (constant + expterm)

    return logp
