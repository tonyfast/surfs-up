# coding: utf-8

# In[1]:

from sklearn.preprocessing import FunctionTransformer
import numpy as np
__all__ = ['RavelPreProcessor']


# In[6]:

RavelPreProcessor = FunctionTransformer(
    lambda x: np.reshape(x, (x.shape[0], np.prod(x.shape[1:]))),
    validate=False,
)


# In[ ]:
