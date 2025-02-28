.. _building-cori:

Cori (NERSC)
============

The `Cori cluster <https://docs.nersc.gov/systems/cori/>`_ is located at NERSC.

If you are new to this system, please see the following resources:

* `GPU nodes <https://docs-dev.nersc.gov/cgpu/access>`__

* `Cori user guide <https://docs.nersc.gov/>`__
* Batch system: `Slurm <https://docs.nersc.gov/jobs/>`__
* `Jupyter service <https://docs.nersc.gov/services/jupyter/>`__
* `Production directories <https://www.nersc.gov/users/storage-and-file-systems/>`__:

  * ``$SCRATCH``: per-user production directory (20TB)
  * ``/global/cscratch1/sd/m3239``: shared production directory for users in the project ``m3239`` (50TB)
  * ``/global/cfs/cdirs/m3239/``: community file system for users in the project ``m3239`` (100TB)

Installation
------------

Use the following commands to download the ImpactX source code and switch to the correct branch:

.. code-block:: bash

   git clone https://github.com/ECP-WarpX/impactx.git $HOME/src/impactx

KNL
^^^

We use the following modules and environments on the system (``$HOME/knl_impactx.profile``).

.. code-block:: bash

   module swap craype-haswell craype-mic-knl
   module swap PrgEnv-intel PrgEnv-gnu
   module load cmake/3.22.1
   module load cray-hdf5-parallel/1.10.5.2
   module load cray-fftw/3.3.8.10
   module load cray-python/3.9.7.1

   export PKG_CONFIG_PATH=$FFTW_DIR/pkgconfig:$PKG_CONFIG_PATH
   export CMAKE_PREFIX_PATH=$HOME/sw/knl/adios2-2.7.1-install:$CMAKE_PREFIX_PATH

   if [ -d "$HOME/sw/knl/venvs/impactx" ]
   then
     source $HOME/sw/knl/venvs/impactx/bin/activate
   fi

   export CXXFLAGS="-march=knl"
   export CFLAGS="-march=knl"

For PICMI and Python workflows, also install a virtual environment:

.. code-block:: bash

   # establish Python dependencies
   python3 -m pip install --user --upgrade pip
   python3 -m pip install --user virtualenv

   python3 -m venv $HOME/sw/knl/venvs/impactx
   source $HOME/sw/knl/venvs/impactx/bin/activate

   python3 -m pip install --upgrade pip
   MPICC="cc -shared" python3 -m pip install --upgrade --no-cache-dir -v mpi4py
   python3 -m pip install --upgrade pytest
   python3 -m pip install --upgrade -r $HOME/src/impactx/requirements.txt
   python3 -m pip install --upgrade -r $HOME/src/impactx/examples/requirements.txt

Haswell
^^^^^^^

We use the following modules and environments on the system (``$HOME/haswell_impactx.profile``).

.. code-block:: bash

   module swap PrgEnv-intel PrgEnv-gnu
   module load cmake/3.22.1
   module load cray-hdf5-parallel/1.10.5.2
   module load cray-fftw/3.3.8.10
   module load cray-python/3.9.7.1

   export PKG_CONFIG_PATH=$FFTW_DIR/pkgconfig:$PKG_CONFIG_PATH
   export CMAKE_PREFIX_PATH=$HOME/sw/haswell/adios2-2.7.1-install:$CMAKE_PREFIX_PATH

   if [ -d "$HOME/sw/haswell/venvs/impactx" ]
   then
     source $HOME/sw/haswell/venvs/impactx/bin/activate
   fi

For PICMI and Python workflows, also install a virtual environment:

.. code-block:: bash

   # establish Python dependencies
   python3 -m pip install --user --upgrade pip
   python3 -m pip install --user virtualenv

   python3 -m venv $HOME/sw/haswell/venvs/impactx
   source $HOME/sw/haswell/venvs/impactx/bin/activate

   python3 -m pip install --upgrade pip
   MPICC="cc -shared" python3 -m pip install --upgrade --no-cache-dir -v mpi4py
   python3 -m pip install --upgrade -r $HOME/src/impactx/requirements.txt

GPU (V100)
^^^^^^^^^^

Cori provides a partition with `18 nodes that include V100 (16 GB) GPUs <https://docs-dev.nersc.gov/cgpu/>`__.
We use the following modules and environments on the system (``$HOME/gpu_impactx.profile``).

.. code-block:: bash

   export proj="m1759"

   module purge
   module load modules
   module load cgpu
   module load esslurm
   module load gcc/8.3.0 cuda/11.4.0 cmake/3.22.1
   module load openmpi

   export CMAKE_PREFIX_PATH=$HOME/sw/cori_gpu/adios2-2.7.1-install:$CMAKE_PREFIX_PATH

   if [ -d "$HOME/sw/cori_gpu/venvs/impactx" ]
   then
     source $HOME/sw/cori_gpu/venvs/impactx/bin/activate
   fi

   # compiler environment hints
   export CC=$(which gcc)
   export CXX=$(which g++)
   export FC=$(which gfortran)
   export CUDACXX=$(which nvcc)
   export CUDAHOSTCXX=$(which g++)

   # optimize CUDA compilation for V100
   export AMREX_CUDA_ARCH=7.0

   # allocate a GPU, e.g. to compile on
   #   10 logical cores (5 physical), 1 GPU
   function getNode() {
       salloc -C gpu -N 1 -t 30 -c 10 --gres=gpu:1 -A $proj
   }

For PICMI and Python workflows, also install a virtual environment:

.. code-block:: bash

   # establish Python dependencies
   python3 -m pip install --user --upgrade pip
   python3 -m pip install --user virtualenv

   python3 -m venv $HOME/sw/cori_gpu/venvs/impactx
   source $HOME/sw/cori_gpu/venvs/impactx/bin/activate

   python3 -m pip install --upgrade pip
   python3 -m pip install --upgrade --no-cache-dir -v mpi4py
   python3 -m pip install --upgrade -r $HOME/src/impactx/requirements.txt

Building ImpactX
----------------

We recommend to store the above lines in individual ``impactx.profile`` files, as suggested above.
If you want to run on either of the three partitions of Cori, open a new terminal, log into Cori and *source* the environment you want to work with:

.. code-block:: bash

   # KNL:
   source $HOME/knl_impactx.profile

   # Haswell:
   #source $HOME/haswell_impactx.profile

   # GPU:
   #source $HOME/gpu_impactx.profile

.. warning::

   Consider that all three Cori partitions are *incompatible*.

   Do not *source* multiple ``...impactx.profile`` files in the same terminal session.
   Open a new terminal and log into Cori again, if you want to switch the targeted Cori partition.

   If you re-submit an already compiled simulation that you ran on another day or in another session, *make sure to source* the corresponding ``...impactx.profile`` again after login!

Then, ``cd`` into the directory ``$HOME/src/impactx`` and use the following commands to compile:

.. code-block:: bash

   cd $HOME/src/impactx
   rm -rf build

   #                       append if you target GPUs:    -DImpactX_COMPUTE=CUDA
   cmake -S . -B build -DImpactX_OPENPMD=ON -DImpactX_DIMS=3
   cmake --build build -j 16


.. _building-cori-tests:

Testing
-------

To run all tests (here on KNL), do:

.. code-block:: bash

   srun -C knl -N 1 -t 30 -q debug ctest --test-dir build --output-on-failure


.. _running-cpp-cori:

Running
-------

Navigate (i.e. ``cd``) into one of the production directories (e.g. ``$SCRATCH``) before executing the instructions below.

KNL
^^^

The batch script below can be used to run a ImpactX simulation on 2 KNL nodes on
the supercomputer Cori at NERSC. Replace descriptions between chevrons ``<>``
by relevant values, for instance ``<job name>`` could be ``laserWakefield``.

Do not forget to first ``source $HOME/knl_impactx.profile`` if you have not done so already for this terminal session.

For PICMI Python runs, the ``<path/to/executable>`` has to read ``python3`` and the ``<input file>`` is the path to your PICMI input script.

.. literalinclude:: ../../../../etc/impactx/cori-nersc/batch_cori.sh
   :language: bash

To run a simulation, copy the lines above to a file ``batch_cori.sh`` and run

.. code-block:: bash

   sbatch batch_cori.sh

to submit the job.

For a 3D simulation with a few (1-4) particles per cell using FDTD Maxwell
solver on Cori KNL for a well load-balanced problem (in our case laser
wakefield acceleration simulation in a boosted frame in the quasi-linear
regime), the following set of parameters provided good performance:

* ``amr.max_grid_size=64`` and ``amr.blocking_factor=64`` so that the size of
  each grid is fixed to ``64**3`` (we are not using load-balancing here).

* **8 MPI ranks per KNL node**, with ``OMP_NUM_THREADS=8`` (that is 64 threads
  per KNL node, i.e. 1 thread per physical core, and 4 cores left to the
  system).

* **2 grids per MPI**, *i.e.*, 16 grids per KNL node.

Haswell
^^^^^^^

The batch script below can be used to run a ImpactX simulation on 1 `Haswell node <https://docs.nersc.gov/systems/cori/>`_ on the supercomputer Cori at NERSC.

Do not forget to first ``source $HOME/haswell_impactx.profile`` if you have not done so already for this terminal session.

.. literalinclude:: ../../../../etc/impactx/cori-nersc/batch_cori_haswell.sh
   :language: bash

To run a simulation, copy the lines above to a file ``batch_cori_haswell.sh`` and
run

.. code-block:: bash

   sbatch batch_cori_haswell.sh

to submit the job.

For a 3D simulation with a few (1-4) particles per cell using FDTD Maxwell
solver on Cori Haswell for a well load-balanced problem (in our case laser
wakefield acceleration simulation in a boosted frame in the quasi-linear
regime), the following set of parameters provided good performance:

* **4 MPI ranks per Haswell node** (2 MPI ranks per `Intel Xeon E5-2698 v3 <https://ark.intel.com/content/www/us/en/ark/products/81060/intel-xeon-processor-e5-2698-v3-40m-cache-2-30-ghz.html>`_), with ``OMP_NUM_THREADS=16`` (which uses `2x hyperthreading <https://docs.nersc.gov/jobs/affinity/>`_)

GPU (V100)
^^^^^^^^^^

Do not forget to first ``source $HOME/gpu_impactx.profile`` if you have not done so already for this terminal session.

Due to the limited amount of GPU development nodes, just request a single node with the above defined ``getNode`` function.
For single-node runs, try to run one grid per GPU.

A multi-node batch script template can be found below:

.. literalinclude:: ../../../../etc/impactx/cori-nersc/batch_cori_gpu.sh
   :language: bash


.. _post-processing-cori:

Post-Processing
---------------

For post-processing, most users use Python via NERSC's `Jupyter service <https://jupyter.nersc.gov>`__ (`Docs <https://docs.nersc.gov/services/jupyter/>`__).

As a one-time preparatory setup, `create your own Conda environment as described in NERSC docs <https://docs.nersc.gov/services/jupyter/#conda-environments-as-kernels>`__.
In this manual, we often use this ``conda create`` line over the officially documented one:

.. code-block:: bash

   conda create -n myenv -c conda-forge python mamba ipykernel ipympl matplotlib numpy pandas yt openpmd-viewer openpmd-api h5py fast-histogram

We then follow the `Customizing Kernels with a Helper Shell Script <https://docs.nersc.gov/services/jupyter/#customizing-kernels-with-a-helper-shell-script>`__ section to finalize the setup of using this conda-environment as a custom Jupyter kernel.

When opening a Jupyter notebook, just select the name you picked for your custom kernel on the top right of the notebook.

Additional software can be installed later on, e.g., in a Jupyter cell using ``!mamba install -c conda-forge ...``.
Software that is not available via conda can be installed via ``!python -m pip install ...``.
