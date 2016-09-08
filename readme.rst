
``surfs-up``
============

Some things I like to use when I am surfing through data.

Installation
------------

``pip install surfs-up``

Basic Usage
===========

.. code:: python

    from dropin import SimplePipeline, RavelPreProcessor, CorrelationModel
    import numpy as np

.. code:: python

    im = np.random.randn(5, 100, 100) > 1

.. code:: python

    RavelPreProcessor.fit_transform(im).sum(axis=1)/100




.. parsed-literal::

    array([ 15.98,  15.94,  15.36,  15.65,  16.26])



.. code:: python

    autocorrelation_result = CorrelationModel(sz=(100,100), s=(150, 150,)).fit_transform(im);

.. code:: python

    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline
    plt.pcolor(np.fft.fftshift(autocorrelation_result[0]));



.. image:: ../readme_files/../readme_6_0.png


--------------

Development
===========

Running the Build and Tests
---------------------------

.. code:: bash

    pip install -r requirements-dev.txt
    python setup.py develop
    watchmedo tricks tricks.yaml

The ``watchmedo`` script will convert your notebooks to scripts and html
files. ``py.test-ipynb`` will test all notebooks matching
``test-*.ipynb``.

Running the docs
----------------

.. code:: bash

    jekyll serve docs -wit

Docs are hosted at ``http://localhost:4000/magical-magic/``.

License
-------

``surfs-up`` is released as free software under the `BSD 3-Clause
license <https://github.com/tonyfast/magical-magic/blob/master/LICENSE>`__.
