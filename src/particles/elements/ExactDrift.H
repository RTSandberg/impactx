/* Copyright 2022-2023 The Regents of the University of California, through Lawrence
 *           Berkeley National Laboratory (subject to receipt of any required
 *           approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * This file is part of ImpactX.
 *
 * Authors: Chad Mitchell, Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#ifndef IMPACTX_EXACTDRIFT_H
#define IMPACTX_EXACTDRIFT_H

#include "particles/ImpactXParticleContainer.H"
#include "mixin/beamoptic.H"
#include "mixin/thick.H"
#include "mixin/nofinalize.H"

#include <AMReX_Extension.H>
#include <AMReX_REAL.H>

#include <cmath>


namespace impactx
{
    struct ExactDrift
    : public elements::BeamOptic<ExactDrift>,
      public elements::Thick,
      public elements::NoFinalize
    {
        static constexpr auto name = "ExactDrift";
        using PType = ImpactXParticleContainer::ParticleType;

        /** A drift using the exact nonlinear transfer map
         *
         * @param ds Segment length in m
         * @param nslice number of slices used for the application of space charge
         */
        ExactDrift( amrex::ParticleReal const ds, int const nslice )
        : Thick(ds, nslice)
        {
        }

        /** Push all particles */
        using BeamOptic::operator();

        /** This is an exactdrift functor, so that a variable of this type can be used like
         *  an exactdrift function.
         *
         * @param p Particle AoS data for positions and cpu/id
         * @param px particle momentum in x
         * @param py particle momentum in y
         * @param pt particle momentum in t
         * @param refpart reference particle
         */
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        void operator() (
                PType& AMREX_RESTRICT p,
                amrex::ParticleReal & AMREX_RESTRICT px,
                amrex::ParticleReal & AMREX_RESTRICT py,
                amrex::ParticleReal & AMREX_RESTRICT pt,
                RefPart const & refpart) const {

            using namespace amrex::literals; // for _rt and _prt

            // access AoS data such as positions and cpu/id
            amrex::ParticleReal const x = p.pos(RealAoS::x);
            amrex::ParticleReal const y = p.pos(RealAoS::y);
            amrex::ParticleReal const t = p.pos(RealAoS::t);

            // initialize output values of momenta
            amrex::ParticleReal pxout = px;
            amrex::ParticleReal pyout = py;
            amrex::ParticleReal ptout = pt;

            // length of the current slice
            amrex::ParticleReal const slice_ds = m_ds / nslice();

            // access reference particle values to find beta, beta*gamma
            amrex::ParticleReal const bet = refpart.beta();
            amrex::ParticleReal const betgam = refpart.beta_gamma();

            // compute the radical in the denominator (= pz):
            amrex::ParticleReal pzden = sqrt(pow(pt-1_prt/bet,2) -
                                1_prt/pow(betgam,2) - pow(px,2) - pow(py,2));

            // advance position and momentum (exact drift)
            p.pos(RealAoS::x) = x + slice_ds * px / pzden;
            // pxout = px;
            p.pos(RealAoS::y) = y + slice_ds * py / pzden;
            // pyout = py;
            p.pos(RealAoS::t) = t - slice_ds * (1_prt/bet +
                               (pt-1_prt/bet)/pzden);
            // ptout = pt;

            // assign updated momenta
            px = pxout;
            py = pyout;
            pt = ptout;

        }

        /** This pushes the reference particle.
         *
         * @param[in,out] refpart reference particle
         */
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        void operator() (RefPart & AMREX_RESTRICT refpart) const {

            using namespace amrex::literals; // for _rt and _prt

            // assign input reference particle values
            amrex::ParticleReal const x = refpart.x;
            amrex::ParticleReal const px = refpart.px;
            amrex::ParticleReal const y = refpart.y;
            amrex::ParticleReal const py = refpart.py;
            amrex::ParticleReal const z = refpart.z;
            amrex::ParticleReal const pz = refpart.pz;
            amrex::ParticleReal const t = refpart.t;
            amrex::ParticleReal const pt = refpart.pt;
            amrex::ParticleReal const s = refpart.s;

            // length of the current slice
            amrex::ParticleReal const slice_ds = m_ds / nslice();

            // assign intermediate parameter
            amrex::ParticleReal const step = slice_ds / sqrt(pow(pt,2)-1.0_prt);

            // advance position and momentum (drift)
            refpart.x = x + step*px;
            refpart.y = y + step*py;
            refpart.z = z + step*pz;
            refpart.t = t - step*pt;

            // advance integrated path length
            refpart.s = s + slice_ds;

        }
    };

} // namespace impactx

#endif // IMPACTX_EXACTDRIFT_H
