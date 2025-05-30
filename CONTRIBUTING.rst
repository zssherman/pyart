Contributor's Guide
===================


The Python ARM Radar Toolkit (Py-ART)
-------------------------------------

The Python ARM Radar Toolkit, Py-ART, is an open source Python module
containing a growing collection of weather radar algorithms and utilities
build on top of the Scientific Python stack and distributed under the
3-Clause BSD license. Py-ART is used by the
`Atmospheric Radiation Measurement (ARM) Climate Research Facility
<http://www.arm.gov>`_ for working with data from a number of precipitation
and cloud radars, but has been designed so that it can be used by others in
the radar and atmospheric communities to examine, processes, and analyze
data from many types of weather radars.


Important Links
---------------

- Official source code repository: https://github.com/ARM-DOE/pyart
- HTML documentation: https://arm-doe.github.io/pyart
- Examples: https://arm-doe.github.io/pyart/examples/
- Mailing List: http://groups.google.com/group/pyart-users/
- Issue Tracker: https://github.com/ARM-DOE/pyart/issues


Citing
------

If you use the Python ARM Radar Toolkit (Py-ART) to prepare a publication
please cite:

    Helmus, J.J. & Collis, S.M., (2016). The Python ARM Radar Toolkit
    (Py-ART), a Library for Working with Weather Radar Data in the Python
    Programming Language. Journal of Open Research Software. 4(1), p.e25.
    DOI: http://doi.org/10.5334/jors.119

Py-ART implements many published scientific methods which should *also* be
cited if you make use of them. Refer to the **References** section in the
documentation of the functions used for information on these citations.


Install
-------

The easiest method for installing Py-ART is to use the conda packages from
the latest release and use Python 3, as Python 2 support ended January 1st,
2020 and many packages including Py-ART no longer support Python 2.
To do this you must download and install
`Anaconda <https://www.anaconda.com/download/#>`_ or
`Miniconda <https://conda.io/miniconda.html>`_.
Then use the following command in a terminal or command prompt to install
the latest version of Py-ART::

    conda install -c conda-forge arm_pyart

To update an older version of Py-ART to the latest release use::

    conda update -c conda-forge arm_pyart

If you are using mamba::

    mamba install -c conda-forge arm_pyart

Resources
---------

Pyart:

- https://github.com/openradar/AMS-Short-Course-on-Open-Source-Radar-Software
- https://github.com/EVS-ATMOS/pyart_short_course
- https://www.youtube.com/watch?v=diiP-Q3bKZw
- https://arm-doe.github.io/pyart/examples/
- https://arm-doe.github.io/pyart/blog.html

Git:

- https://git-scm.com/book/en/v2


Code Style
----------

Py-ART uses pre-commit for linting, which applies a variety of pep8 and other code style rules.

For more on pep8 style:

- https://www.python.org/dev/peps/pep-0008/

To install pre-commit hooks for the Py-ART repo::

        pre-commit install

Afterwards, pre-commit will run on every commit to the repository. It will re-format files as neccessary.


Python File Setup
-----------------

In a new .py file, the top of the code should have a brief introduction to
the module.

An example:

.. code-block:: python

        """
        Retrieval of VADs from a radar object.

        """

Following the introduction code, modules are then added. To follow pep8
standards, modules should be added in the order of:

        1. Standard library imports.
        2. Related third party imports.
        3. Local application/library specific imports.

For example:

.. code-block:: python

        import glob
        import os

        import numpy as np
        import numpy.ma as ma
        from scipy.interpolate import interp1d

        from ..core import HorizontalWindProfile

Following the main function def line, but before the code within it, a doc
string is needed to explain arguments, returns, references if needed, and
other helpful information. These documentation standards follow the NumPy
documentation style.

For more on the NumPy documentation style:

- https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard

An example:

.. code-block:: python

        def velocity_azimuth_display(
            radar, velocity=None, z_want=None, valid_ray_min=16,
            gatefilter=False, window=2):

            """
   	    Velocity azimuth display.

            Parameters
            ----------
            radar : Radar
                Radar object used.
            velocity : string
                Velocity field to use for VAD calculation.
                If None, the default velocity field will be used.

            Other Parameters
            ----------------
            z_want : array
                Height array user would like for the VAD
                calculation. None will result in a z_want of
        	np.linspace and use of _inverse_dist_squared
        	and _Average1D functions. Note, height must have
        	same shape as expected u_wind and v_wind if user
        	provides z_want.
    	    valid_ray_min : int
        	Amount of rays required to include that level in
        	the VAD calculation.
            gatefilter : GateFilter
        	Used to correct the velocity field before its use
        	in the VAD calculation. Uses Py-ART's region dealiaser.
    	    window : int
        	Value to use for window calculation in _Averag1D
        	function.

            Returns
            -------
    	    height : array
        	Heights in meters above sea level at which horizontal
                winds were sampled.
    	    speed : array
        	Horizontal wind speed in meters per second at each height.
    	    direction : array
        	Horizontal wind direction in degrees at each height.
    	    u_wind : array
        	U-wind mean in meters per second.
    	    v_wind : array
        	V-wind mean in meters per second.

    	    Reference
    	    ----------
    	    K. A. Browning and R. Wexler, 1968: The Determination
    	    of Kinematic Properties of a Wind Field Using Doppler
	    Radar. J. Appl. Meteor., 7, 105–113

    	    """

As seen, each argument has what type of object it is, an explanation of
what it is, mention of units, and if an argument has a default value, a
statement of what that default value is and why.

Private or smaller functions and classes can have a single line explanation.

An example:

.. code-block:: python

        def u_wind(self):
        """ U component of horizontal wind in meters per second. """


Testing
-------

When adding a new function to pyart it is important to add your function to
the __init__.py file under the corresponding pyart folder.

Create a test for your function and have assert from numpy testing test the
known values to the calculated values. If changes are made in the future to
pyart, pytest will use the test created to see if the function is still valid and
produces the same values. It works that, it takes known values that are
obtained from the function, and when pytest is ran, it takes the test
function and reruns the function and compares the results to the original.

An example continuing to use the VAD code above:

.. code-block:: python

        def test_vad():
            test_radar = pyart.testing.make_target_radar()
            height = np.arange(0, 1000, 200)
            speed = np.ones_like(height) * 5
            direction = np.array([0, 90, 180, 270, 45])
            profile = pyart.core.HorizontalWindProfile(
                height, speed, direction)
            sim_vel = pyart.util.simulated_vel_from_profile(
                test_radar, profile)

            test_radar.add_field('velocity', sim_vel,
                                 replace_existing=True)

            velocity = 'velocity'
            z_start = 0
            z_end = 10
            z_count = 5

            vad_height = ([0., 2.5, 5., 7.5, 10.])
            vad_speed = ([4.98665725, 4.94020686, 4.88107152,
                          4.81939374, 4.75851962])
            vad_direction = ([359.84659496, 359.30240553, 358.58658589,
                              357.81073051, 357.01353486])
            u_wind = ([0.01335138, 0.06014712, 0.12039762,
                       0.18410404, 0.24791911])
            v_wind = ([-4.98663937, -4.9398407, -4.87958641,
                       -4.81587601, -4.75205693])

            vad = pyart.retrieve.velocity_azimuth_display(test_radar,
                                                          velocity,
                                                          z_start, z_end,
                                                          z_count)

            assert_almost_equal(vad.height, vad_height, 3)
            assert_almost_equal(vad.speed, vad_speed, 3)
            assert_almost_equal(vad.direction, vad_direction, 3)
            assert_almost_equal(vad.u_wind, u_wind, 3)
            assert_almost_equal(vad.v_wind, v_wind, 3)

Pytest is used to run unit tests in pyart.

It is recommended to install pyart in “editable” mode for pytest testing.
From within the main pyart directory::

        pip install -e .

This lets you change your source code and rerun tests at will.

To install pytest::

        conda install pytest

To run all tests in pyart with pytest from outside the pyart directory::

        pytest --pyargs pyart

All test with increase verbosity::

        pytest -v

Just one file::

        pytest filename

Note: When an example shows filename as such::

        pytest filename

filename is the filename and location, such as::

        pytest /home/user/pyart/pyart/io/tests/test_cfradial.py

Relative paths can also be used::

        cd pyart
        pytest ./pyart/retrieve/tests/test_vad.py

For more on pytest:

- https://docs.pytest.org/en/latest/

Examples
--------

When creating and contributing new examples to Py-ART, we used the open-radar-data
package for example files.

To install open-radar-data::

        conda install -c conda-forge open-radar-data

To see what files exist in open-radar-data to use for an example script:

.. code-block:: python

   from open_radar_data import DATASETS

   print(DATASETS.registry_files)

To grab a file to use for an example, use:

.. code-block:: python

   files = DATASETS.fetch('gucxprecipradarcmacppiS2.c1.20220314.025840.nc')

If there isn't a file that fits the example being created, feel free to open a PR at
https://github.com/openradar/open-radar-data

GitHub
------

When contributing to pyart, the changes created should be in a new branch
under your forked repository. Let's say the user is adding a new map display.
Instead of creating that new function in your main branch. Create a new
branch called ‘cartopy_map’. If everything checks out and the admin
accepts the pull request, you can then merge the main branch and
cartopy_map branch.

To delete a branch both locally and remotely, if done with it::

		git push origin --delete <branch_name>
		git branch -d <branch_name>

or in this case::

		git push origin --delete cartopy_map
		git branch -d cartopy_map


To create a new branch::

                git checkout -b <branch_name>

If you have a branch with changes that have not been added to a pull request
but you would like to start a new branch with a different task in mind. It
is recommended that your new branch is based on your main. First::

                git checkout main
Then::

                git checkout -b <branch_name>

This way, your new branch is not a combination of your other task branch and
the new task branch, but is based on the original main branch.

Typing `git status` will not only inform the user of what files have been
modified and untracked, it will also inform the user of which branch they
are currently on.

To switch between branches, simply type::

		git checkout <branch_name>

When commiting to GitHub, start the statement with a acronym such as
‘ADD:’ depending on what your commiting, could be ‘MAINT:’ or
‘BUG:’ or more. Then following should be a short statement such as
“ADD: Adding cartopy map display.”, but after the short statement, before
finishing the quotations, hit enter and in your terminal you can then type
a more in depth description on what your commiting.

A set of recommended acronymns can be found at:

- https://docs.scipy.org/doc/numpy/dev/gitwash/development_workflow.html

If you would like to type your commit in the terminal and skip the default
editor::

	git commit -m "STY: Removing whitespace from vad.py pep8."

To use the default editor(in Linux, usually VIM), simply type::

	git commit

One thing to keep in mind is before doing a pull request, update your
branches with the original upstream repository.

This could be done by::

	git fetch upstream

After fetching, a git merge is needed to pull in the changes.

This is done by::

        git merge upstream/main

To prevent a merge commit::

        git merge --ff-only upstream/main

After creating a pull request through GitHub, GitHub Actions will determine
if the code past all checks for multiple Python versions as well as Operating
Systems. If the code fails either tests, as the pull request sits, we will
provide guidance to make changes to fix the code. Once the code is fixed and
committed to GitHub, GitHub Actions will automatically rerun. A documentation
build will also run to update Py-ART's documentation with the new changes.
Code coverage will also provided statistics if your new code is properly
unit tested. PEP8 check will also run with changes to fix to keep the code
to PEP8 standards.
