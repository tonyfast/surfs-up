# coding: utf-8

# In[4]:

from dropin.pipeline import SimplePipeline
from whatever import _X
from toolz.curried import first
from sklearn.preprocessing import FunctionTransformer

from pandas import np

__version__ = "0.0.1"

__all__ = ['normalized_correlation']


# In[65]:

weiner_khinchin_auto_correlation = (
    _X() | np.fft.fftn | np.abs | (lambda x: x**2) | np.fft.ifftn | np.real
)[np.around](decimals=13)


# In[193]:

def auto_correlation(im, **kwargs):
    windowed = weiner_khinchin_auto_correlation.copy()
    for i, token in enumerate(windowed._tokens):
        if first(token) in (
            np.fft.fftpack.fftn, np.fft.fftpack.ifftn,
        ):
            windowed._tokens[i][2] = kwargs
    return windowed.value(im)


# In[2]:

def cross_correlation(X, Y, **kwargs):
    fft_based_cross_correlation = _X(
        np.fft.fftn(X, **kwargs), np.fft.fftn(Y, **kwargs).conj()
    )[np.multiply][np.fft.ifftn](**kwargs)[np.real][np.around](decimals=13)
    return fft_based_cross_correlation.value()


# In[3]:

def apply_denominator(X, s=(10, 10,), cutoff=(20, 20,)):
    return np.apply_over_axes(
        lambda x, y: np.divide(x, auto_correlation(np.ones(s), s=cutoff)),
        X, axes=[0]
    )


# In[328]:

correlation_model = FunctionTransformer(
    auto_correlation, validate=False, kw_args=dict(s=(20, 20,))
)

normalization = FunctionTransformer(
    apply_denominator, validate=False, kw_args={}
)


normalized_correlation = SimplePipeline(
    [[correlation_model], [normalization]], n_jobs=1)
