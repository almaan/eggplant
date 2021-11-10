Installation
============

GitHub
~~~~~~

To install ``eggplant``, first create a new directory into which you'd like to
clone the github reposiory conatining the source code, or alternatively use an
already existing folder. In your terminal emulator, change directory into the
designated folder (``PARENT_DIR``) and then clone the repository:

.. code-block:: console

   cd PARENT_DIR
   git clone git@github.com:almaan/eggplant.git

Next, enter the root folder of the repository:

.. code-block:: console

   cd eggplant
 
If you're using ``conda`` or any flavor of it, we provide a minimal functioning
environment in ``conda/eggplant.yaml``. You can create and load this environment
by doing:

.. code-block:: console

   # activate conda environment
   conda active
   # create a conda enviroment from our provided file 
   conda env create -f conda/eggplant.yaml
   # activate the environment
   conda activate eggplant

To install ``eggplant`` we will then use the ``setup.py`` file:

.. code-block:: console

   cd eggplant
   python3 ./setup.py install --user

To test that your installation was successful you could try:

.. code-block:: console

   python3 -c "import eggplant as eg; print(eg.__version__)"

This should print the version number of eggplant. If you want to do some even
heavier testing, you could also run the unit tests, done by executing:

.. code-block:: console

   python3 -m unittest discover

PIP
~~~
You can also install ``eggplant`` via ``pip``, by executing the command:

.. code-block:: console

                python3 -m pip spatial-eggplant

However, we **recommend** to use the GitHub alternative, as this will give you
the most up-to-date version.
