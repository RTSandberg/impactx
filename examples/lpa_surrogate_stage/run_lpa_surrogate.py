#!/usr/bin/env python3
#
# Copyright 2022-2023 ImpactX contributors
# Authors: Ryan Sandberg, Axel Huebl
# License: BSD-3-Clause-LBNL
#
# -*- coding: utf-8 -*-

import numpy as np

try:
    import cupy as cp

    cupy_available = True
except ImportError:
    cupy_available = False

import amrex
from impactx import Config, ImpactX, RefPart, distribution, elements, transformation

sim = ImpactX()

# set numerical parameters and IO control
sim.particle_shape = 2  # B-spline order
sim.space_charge = False
# sim.diagnostics = False  # benchmarking
sim.slice_step_diagnostics = True

# domain decomposition & space charge mesh
sim.init_grids()

# load a 2 GeV electron beam with an initial
# unnormalized rms emittance of 2 nm
# !! want 0.5 nm
energy_MeV = 2.0e3  # reference energy
bunch_charge_C = 1.0e-10  # used with space charge
npart = 10000  # number of macro particles

#   reference particle
ref = sim.particle_container().ref_particle()
ref.set_charge_qe(-1.0).set_mass_MeV(0.510998950).set_energy_MeV(energy_MeV)

#   particle bunch
distr = distribution.Waterbag(
    sigmaX=3.9984884770e-5,
    sigmaY=3.9984884770e-5,
    sigmaT=1.0e-3,
    sigmaPx=2.6623538760e-5,
    sigmaPy=2.6623538760e-5,
    sigmaPt=2.0e-3,
    muxpx=-0.846574929020762,
    muypy=0.846574929020762,
    mutpt=0.0,
)
sim.add_particles(bunch_charge_C, distr, npart)

# number of slices per ds in each lattice element
ns = 25



# build a custom, Pythonic beam optical element
def surrogate_plugin(pge, pti, refpart, reference_particle_z0):
    """This pushes the beam particles ... .

    Relative to the reference particle.

    :param pti: particle iterator for the current tile or box
    :param refpart: the reference particle
    """
    # CPU/GPU logic
    if Config.have_gpu:
        if cupy_available:
            array = cp.array
        else:
            print("Warning: GPU found but cupy not available! Try managed...")
            array = np.array
        if Config.gpu_backend == "SYCL":
            print("Warning: SYCL GPU backend not yet implemented for Python")

    else:
        array = np.array

    # load my neural net
    
    # apply s-to-t transform
    # need to get particle container pc
    coordinate_transformation(pc, impactx.TransformationDirection.to_fixed_t)

    # transform to neural net coordinates
    # (here we assume that we come out of a drift section or otherwise don't need to modify x,y)
    # subtract
    # normalize to model coordinates
    ## subtract means, divide by stds
    # apply neural net
    ## unnormalize
    ## multiply by stds, add means
    # apply t-to-s transform
    coordinate_transformation(pc, impactx.TransformationDirection.to_fixed_s)
    # return



def ref_surrogate(pge, refpart, reference_particle_z0):
    """This pushes the reference particle.

    :param refpart: reference particle
    """
    #  assign input reference particle values
    x = refpart.x
    px = refpart.px
    y = refpart.y
    py = refpart.py
    z = refpart.z
    pz = refpart.pz
    t = refpart.t
    pt = refpart.pt
    s = refpart.s

    #
    ref_x = ref_y = 0
    ref_z = reference_particle_z0
    ref_gamma = -refpart.pt

    # apply neural net
    ## normalize
    ## apply
    ## unnormalizes

    # advance position and momentum (drift)
    refpart.x = 
    refpart.y = 
    refpart.z = z + step * pz
    refpart.t = t - step * pt

    # advance integrated path length
    refpart.s = s + stage_length


pge1 = elements.Programmable()
pge1.nslice = ns
pge1.beam_particles = lambda pti, refpart: surrogate_plugin(pge1, pti, refpart)
pge1.ref_particle = lambda refpart: ref_surrogate(pge1, refpart)
pge1.ds = 0.25


# design the accelerator lattice
surrogate_lattice = [
    pge1
]
# assign a fodo segment
sim.lattice.extend(surrogate_lattice)

# run simulation
sim.evolve()

# clean shutdown
del sim
amrex.finalize()
