/* Copyright 2022-2023 The Regents of the University of California, through Lawrence
 *           Berkeley National Laboratory (subject to receipt of any required
 *           approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * This file is part of ImpactX.
 *
 * Authors: Chad Mitchell, Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#ifndef IMPACTX_TO_FIXED_S_H
#define IMPACTX_TO_FIXED_S_H

#include "particles/ImpactXParticleContainer.H"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <cmath>


namespace impactx
{
namespace transformation
{
    struct ToFixedS
    {
        using PType = ImpactXParticleContainer::ParticleType;

        /** Transformation of particles from fixed time t to fixed location s.
         *
         * At fixed s, each particle is represented by phase space
         *  coordinates (x,y,t,px,py,pt).
         *
         * @param pzd Design value of pz/mc = beta*gamma.
         */
        ToFixedS (amrex::ParticleReal const pzd)
        : m_pzd(pzd)
        {
        }

        /** This is a t-to-s map, so that a variable of this type can be used like a
         *  t-to-s function.
         *
         * @param[inout] p Particle AoS data for positions and cpu/id
         * @param[inout] px particle momentum in x
         * @param[inout] py particle momentum in y
         * @param[inout] pz particle momentum in z (in), in t (out)
         */
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        void operator() (
            PType& p,
            amrex::ParticleReal & px,
            amrex::ParticleReal & py,
            amrex::ParticleReal & pz) const
        {
            using namespace amrex::literals;

            // access AoS data such as positions and cpu/id
            amrex::ParticleReal const x = p.pos(RealAoS::x);
            amrex::ParticleReal const y = p.pos(RealAoS::y);
            amrex::ParticleReal const z = p.pos(RealAoS::z);

            // compute value of reference ptd = -gamma
            amrex::ParticleReal const argd = 1.0_prt + pow(m_pzd, 2);
            AMREX_ASSERT_WITH_MESSAGE(argd > 0.0_prt, "invalid ptd arg (<=0)");
            amrex::ParticleReal const ptdf = argd > 0.0_prt ? -sqrt(argd) : -1.0_prt;

            // transform momenta to dynamic units (e.g., so that momenta are
            // normalized by mc):
            px = px * m_pzd;
            py = py * m_pzd;
            pz = pz * m_pzd;

            // compute value of particle pt = -gamma
            amrex::ParticleReal const arg = 1.0_prt + pow(m_pzd + pz, 2) + pow(px, 2) + pow(py, 2);
            AMREX_ASSERT_WITH_MESSAGE(arg > 0.0_prt, "invalid pt arg (<=0)");
            amrex::ParticleReal const ptf = arg > 0.0_prt ? -sqrt(arg) : -1.0_prt;

            // transform position and momentum (from fixed t to fixed s)
            p.pos(RealAoS::x) = x - px * z / (m_pzd + pz);
            // px = px;
            p.pos(RealAoS::y) = y - py * z / (m_pzd + pz);
            // py = py;
            p.pos(RealAoS::t) = ptf * z / (m_pzd + pz);
            auto & pt = pz;  // We store pt in the same memory slot as pz.
            pt = ptf - ptdf;

            // transform momenta to static units (eg, so that momenta are
            // normalized by pzd):
            px = px / m_pzd;
            py = py / m_pzd;
            pt = pz / m_pzd;
        }

    private:
        amrex::ParticleReal m_pzd;  ///< Design value of pz/mc = beta*gamma.
    };

} // namespace transformation
} // namespace impactx

#endif // IMPACTX_TO_FIXED_S_H
