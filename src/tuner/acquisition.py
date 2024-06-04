import warnings
from scipy.stats import norm

def ei(x, gp, y_min, xi=0):
    r"""Calculate Expected Improvement acquisition function.

    Calculated as

    .. math::
            \text{EI}(x) = (\mu(x) - y_{\text{max}} - \xi) \Phi\left(
                \frac{-\mu(x) + y_{\text{max}} + \xi }{\sigma(x)} \right)
                  + \sigma(x) \phi\left(
                    \frac{-\mu(x) + y_{\text{max}} + \xi }{\sigma(x)} \right)

    where :math:`\Phi` is the CDF and :math:`\phi` the PDF of the normal
    distribution.

    Parameters
    ----------
    x : np.ndarray
        Parameters to evaluate the function at.

    gp : sklearn.gaussian_process.GaussianProcessRegressor
        A gaussian process regressor modelling the target function based on
        previous observations.
    
    y_max : number
        Highest found value of the target function.
        
    xi : float, positive
        Governs the exploration/exploitation tradeoff. Lower prefers
        exploitation, higher prefers exploration.

    Returns
    -------
    Values of the acquisition function
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean, std = gp.predict(x, return_std=True)

    a = mean - y_min - xi
    z = -a / std

    return a * norm.cdf(z) + std * norm.pdf(z) # how much lesser *on average* is f(x) when compared to current found min
