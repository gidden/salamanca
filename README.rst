`salamanca` - A Python library for working with socio-economic indicators
=========================================================================

.. image:: https://circleci.com/gh/gidden/salamanca.svg?style=shield&circle-token=:circle-token
    :target: https://circleci.com/gh/gidden/salamanca

.. image:: https://coveralls.io/repos/github/gidden/salamanca/badge.svg?branch=master
    :target: https://coveralls.io/github/gidden/salamanca?branch=master
   
**Please note that salamanca is still in early developmental stages, thus all interfaces are subject to change.**

Documentation
-------------

All documentation can be found at http://mattgidden.com/salamanca

Install
-------

From Source
***********

Installing from source is as easy as

.. code-block:: bash

    pip install -r requirements.txt && python setup.py install

Build the Docs
--------------

Requirements
************

- `cloud_sptheme`
- `numpydoc`
- `nbsphinx`
- `sphinxcontrib-programoutput`
- `sphinxcontrib-exceltable`

Build and Serve
***************

.. code-block:: bash

    cd doc
    make html

Then point you browser to `http://127.0.0.1:8000/`.

License
-------

Licensed under Apache 2.0. See the LICENSE file for more information
