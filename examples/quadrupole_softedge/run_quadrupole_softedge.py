#!/usr/bin/env python3
#
# Copyright 2022-2023 ImpactX contributors
# Authors: Axel Huebl, Chad Mitchell
# License: BSD-3-Clause-LBNL
#
# -*- coding: utf-8 -*-

import amrex
from impactx import ImpactX, RefPart, distribution, elements

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
energy_MeV = 2.0e3  # reference energy
bunch_charge_C = 1.0e-9  # used with space charge
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

# add beam diagnostics
monitor = elements.BeamMonitor("monitor", backend="h5")

# design the accelerator lattice
ns = 1  # number of slices per ds in the element

quad1 = elements.SoftQuadrupole(
    ds=1.0,
    gscale=1.0,
    cos_coefficients=[2],
    sin_coefficients=[0],
    mapsteps=400,
    nslice=ns,
)

quad2 = elements.SoftQuadrupole(
    ds=1.0,
    gscale=-1.0,
    cos_coefficients=[2],
    sin_coefficients=[0],
    mapsteps=200,
    nslice=ns,
)

drift1 = elements.Drift(ds=0.25, nslice=ns)
drift2 = elements.Drift(ds=0.5, nslice=ns)

# assign a fodo segment
sim.lattice.extend([monitor, drift1, quad1, drift2, quad2, drift1, monitor])

# run simulation
sim.evolve()

# clean shutdown
del sim
amrex.finalize()
