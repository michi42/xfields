// copyright ################################# //
// This file is part of the Xfields Package.   //
// Copyright (c) CERN, 2021.                   //
// ########################################### //

#ifndef XFIELDS_BEAMBEAM_MULTIBUNCH_2D_H
#define XFIELDS_BEAMBEAM_MULTIBUNCH_2D_H

#include "xtrack/headers/track.h"
#include "xfields/fieldmaps/bigaussian_src/bigaussian.h"


GPUFUN
void BeamBeamBiGaussianMultibunch2D_track_local_particle(
        BeamBeamBiGaussianMultibunch2DData el, LocalParticle* part0){

    double const scale_strength = BeamBeamBiGaussianMultibunch2DData_get_scale_strength(el);

    double const zeta_offset = BeamBeamBiGaussianMultibunch2DData_get_zeta_offset(el);
    double const zeta_match_tol = BeamBeamBiGaussianMultibunch2DData_get_zeta_match_tol(el);
    double const zeta_period = BeamBeamBiGaussianMultibunch2DData_get_zeta_period(el);

    double const other_beam_q0 = scale_strength*BeamBeamBiGaussianMultibunch2DData_get_other_beam_q0(el);
    double const other_beam_beta0 = BeamBeamBiGaussianMultibunch2DData_get_other_beam_beta0(el);
    double const other_beam_gamma0 = BeamBeamBiGaussianMultibunch2DData_get_other_beam_gamma0(el);

    double const min_sigma_diff = BeamBeamBiGaussianMultibunch2DData_get_min_sigma_diff(el);

    int64_t const num_other_bunches = BeamBeamBiGaussianMultibunch2DData_get_num_other_bunches(el);

    START_PER_PARTICLE_BLOCK(part0, part);
        double const x = LocalParticle_get_x(part);
        double const y = LocalParticle_get_y(part);
        double const zeta = LocalParticle_get_zeta(part);
        double const part_q0 = LocalParticle_get_q0(part);
        double const part_mass0 = LocalParticle_get_mass0(part);
        double const part_chi = LocalParticle_get_chi(part);
        double const part_beta0 = LocalParticle_get_beta0(part);
        double const part_gamma0 = LocalParticle_get_gamma0(part);

        // This particle (bunch) at `zeta` encounters the opposing bunch
        // located at `zeta + zeta_offset`. Find the closest matching bunch.
        // If `zeta_period` > 0 the bunch-label axis is periodic (circular
        // machine): the distance is evaluated modulo the period, so encounter
        // offsets that wrap around the ring still find their partner.
        // The opposing bunches are stored SORTED in zeta (enforced by
        // `update_from_other_beam`), so the partner is found by binary search:
        // the nearest (mod period) bunch is either a linear neighbour of the
        // folded target or, across the wrap, one of the two ends.
        double const target_zeta = zeta + zeta_offset;
        int64_t i_match = -1;
        double best_dist = zeta_match_tol;
        if (num_other_bunches > 0){
            double tt = target_zeta;
            if (zeta_period > 0.){
                double const z_first = BeamBeamBiGaussianMultibunch2DData_get_other_beam_zeta(el, 0);
                double const z_last = BeamBeamBiGaussianMultibunch2DData_get_other_beam_zeta(
                                                el, num_other_bunches - 1);
                double const z_mid = 0.5 * (z_first + z_last);
                tt -= zeta_period * round((tt - z_mid) / zeta_period);
            }
            int64_t lo = 0;                      // lower bound: first z >= tt
            int64_t hi = num_other_bunches;
            while (lo < hi){
                int64_t const mid = (lo + hi) / 2;
                if (BeamBeamBiGaussianMultibunch2DData_get_other_beam_zeta(el, mid) < tt){
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            int64_t const cand[4] = {lo - 1, lo, 0, num_other_bunches - 1};
            for (int cc = 0; cc < 4; cc++){
                int64_t const jj = cand[cc];
                if (jj < 0 || jj >= num_other_bunches) continue;
                double dist = BeamBeamBiGaussianMultibunch2DData_get_other_beam_zeta(el, jj)
                              - target_zeta;
                if (zeta_period > 0.){
                    dist -= zeta_period * round(dist / zeta_period);
                }
                dist = fabs(dist);
                if (dist <= best_dist){
                    best_dist = dist;
                    i_match = jj;
                }
            }
        }

        if (i_match < 0){
            // No opposing bunch at the encounter position -> no kick
            continue;
        }

        double const other_beam_shift_x = BeamBeamBiGaussianMultibunch2DData_get_other_beam_x(el, i_match);
        double const other_beam_shift_y = BeamBeamBiGaussianMultibunch2DData_get_other_beam_y(el, i_match);
        double const other_beam_num_particles =
            BeamBeamBiGaussianMultibunch2DData_get_other_beam_num_particles(el, i_match);

        // Per-bunch transverse size from normalized emittance and beta:
        //   sigma = sqrt(beta_twiss * nemitt / gamma_rel)
        double const nemitt_x = BeamBeamBiGaussianMultibunch2DData_get_other_beam_nemitt_x(el, i_match);
        double const nemitt_y = BeamBeamBiGaussianMultibunch2DData_get_other_beam_nemitt_y(el, i_match);
        double const betx = BeamBeamBiGaussianMultibunch2DData_get_other_beam_betx(el, i_match);
        double const bety = BeamBeamBiGaussianMultibunch2DData_get_other_beam_bety(el, i_match);

        double const sigma_x = sqrt(betx * nemitt_x / other_beam_gamma0);
        double const sigma_y = sqrt(bety * nemitt_y / other_beam_gamma0);

        double const x_bar = x - other_beam_shift_x;
        double const y_bar = y - other_beam_shift_y;

        // Get transverse fields
        double Ex, Ey; // Ex = -dphi/dx, Ey = -dphi/dy
        get_Ex_Ey_gauss(x_bar, y_bar,
            sigma_x, sigma_y,
            min_sigma_diff,
            &Ex, &Ey);

        const double charge_mass_ratio = part_chi*QELEM*part_q0
                    /(part_mass0*QELEM/(C_LIGHT*C_LIGHT));
        const double factor = (charge_mass_ratio
                    * other_beam_num_particles * other_beam_q0 * QELEM
                    / (part_gamma0*part_beta0*C_LIGHT*C_LIGHT)
                    * (1+other_beam_beta0 * part_beta0)
                    / (other_beam_beta0 + part_beta0));

        double const dpx = factor * Ex;
        double const dpy = factor * Ey;

        LocalParticle_add_to_px(part, dpx);
        LocalParticle_add_to_py(part, dpy);
    END_PER_PARTICLE_BLOCK;
}

#endif
