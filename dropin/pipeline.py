# coding: utf-8

# In[1]:

from whatever import _X, callables
from toolz.curried import identity
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.pipeline import make_union, Pipeline
from sklearn.ensemble import VotingClassifier
__all__ = ['SimplePipeline']


# In[2]:

class SimplePipeline(Pipeline):
    """Build a pipeline from a list."""

    def __init__(self, pipeline, n_jobs=4):
        pipeline = list(pipeline)

        if _X(pipeline[-1]) * (
            lambda x: isinstance(x, (
                RegressorMixin, ClassifierMixin,
            ))
        ) | all > identity:
            pipeline[-1] = VotingClassifier(
                _X(pipeline[-1]) * [str, identity] > list,
            )

        pipeline = _X(pipeline) * callables.Dispatch([
            [VotingClassifier, identity],
        ], lambda m: make_union(*m).set_params(n_jobs=n_jobs)) > list

        self.pipeline = pipeline
        self.n_jobs = n_jobs
        super().__init__(
            _X(pipeline) * [str, identity] > list
        )


# In[ ]:
