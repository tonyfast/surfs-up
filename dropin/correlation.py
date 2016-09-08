# coding: utf-8

# In[19]:

from dropin.pipeline import SimplePipeline
from whatever import _X
from toolz.curried import first
from sklearn.preprocessing import FunctionTransformer

import numpy as np

__version__ = "0.0.1"

__all__ = ['CorrelationModel']


# In[20]:

weiner_khinchin_auto_correlation = (
    _X() | np.fft.fftn | np.abs | (lambda x: x**2) | np.fft.ifftn | np.real
)[np.around](decimals=13)


# In[21]:

def auto_correlation(im, **kwargs):
    windowed = weiner_khinchin_auto_correlation.copy()
    for i, token in enumerate(windowed._tokens):
        if first(token) in (
            np.fft.fftpack.fftn, np.fft.fftpack.ifftn,
        ):
            windowed._tokens[i][2] = kwargs
    return windowed.value(im)


# In[22]:

def cross_correlation(X, Y, **kwargs):
    fft_based_cross_correlation = _X(
        np.fft.fftn(X, **kwargs), np.fft.fftn(Y, **kwargs).conj()
    )[np.multiply][np.fft.ifftn](**kwargs)[np.real][np.around](decimals=13)
    return fft_based_cross_correlation.value()


# In[23]:

def apply_denominator(X, sz=(10, 10,), s=(20, 20,)):
    return np.apply_over_axes(
        lambda x, y: np.divide(x, auto_correlation(np.ones(sz), s=s)),
        X, axes=[0]
    )


# In[30]:

def CorrelationModel(sz, **kwargs):
    auto_correlation_model = FunctionTransformer(
        auto_correlation, validate=False, kw_args={**kwargs, 'axes': _X(sz).len.add(1).range(1).list.value()}
    )

    normalization = FunctionTransformer(
        apply_denominator, validate=False, kw_args={'sz': sz, **kwargs}
    )

    return SimplePipeline([
        #         [FunctionTransformer(_X().eq(0).compose, validate=False)],
        [auto_correlation_model], [normalization]
    ], n_jobs=1)


# In[ ]:
