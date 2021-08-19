Installation
============

GitHub
~~~~~~

To install ``eggplant``, first create a new directory into which you'd like to
clone the github reposiory conatining the source code, or alternatively use an
already existing folder. In the terminal, change directory into the designated
folder (``PARENT_DIR``) and then clone the repo:

.. code-block:: console

   cd PARENT_DIR
   git clone git@github.com:almaan/eggplant.git

Next install enter the _eggplant_ folder and install ``eggplant`` using ``setup.py``:

.. code-block:: console

   cd eggplant
   python3 ./setup.py install --user

To test that your installation was successful you could try:

.. code-block:: console

   python3 -c "import eggplant as eg; print(eg.__version__)"

This should print the version number of eggplant.
