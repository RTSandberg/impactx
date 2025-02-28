/* Copyright 2022-2023 The Regents of the University of California, through Lawrence
 *           Berkeley National Laboratory (subject to receipt of any required
 *           approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * This file is part of ImpactX.
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#ifndef IMPACTX_PUSH_ALL_H
#define IMPACTX_PUSH_ALL_H

#include "particles/ImpactXParticleContainer.H"

#include <AMReX_BLProfiler.H>


namespace impactx
{
    /** Push all particles in a particle container.
     *
     * This element pushes first the reference particle, then all other particles.
     * All particles are pushed independently with the same logic.
     * Particles are relative to the reference particle.
     *
     * @param[in,out] pc particle container to push
     * @param[in,out] element the beamline element
     * @param[in] step global step for diagnostics
     * @param[in] omp_parallel allow threading via OpenMP for the particle iterator loop (note: if OMP backend is active)
     */
    template<typename T_Element>
    void push_all (
            ImpactXParticleContainer & pc,
            T_Element & element,
            [[maybe_unused]] int step,
            [[maybe_unused]] bool omp_parallel = true
    )
    {
        // preparing to access reference particle data: RefPart
        RefPart & ref_part = pc.GetRefParticle();

        // push reference particle in global coordinates
        {
            BL_PROFILE("impactx::Push::RefPart");
            element(ref_part);
        }

        // loop over refinement levels
        int const nLevel = pc.finestLevel();
        for (int lev = 0; lev <= nLevel; ++lev)
        {
            // loop over all particle boxes
            using ParIt = ImpactXParticleContainer::iterator;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion() && omp_parallel)
#endif
            for (ParIt pti(pc, lev); pti.isValid(); ++pti) {
                // push beam particles relative to reference particle
                element(pti, ref_part);
            } // end loop over all particle boxes
        } // env mesh-refinement level loop
    }

} // namespace impactx

#endif // IMPACTX_PUSH_ALL_H
