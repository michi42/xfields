# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import numpy as np

import xobjects as xo
import xtrack as xt


class BeamBeamBiGaussianMultibunch2D(xt.BeamElement):

    """
    2D (transverse) beam-beam element for multi-bunch beams in the
    soft-Gaussian approximation.

    The opposing beam is described as a set of bunches, each one represented
    by a single macroparticle holding the bunch centroid (``x``, ``y``), its
    longitudinal position (``zeta``) and its population (number of real
    charges). The transverse charge distribution of every opposing bunch is
    assumed Gaussian, with a per-bunch size derived from its normalized
    emittances and beta functions::

        sigma = sqrt(beta_twiss * nemitt / gamma0)

    where ``gamma0`` is the relativistic gamma of the opposing beam.

    During tracking, a particle (bunch) of this beam located at ``zeta``
    interacts with the opposing bunch located at ``zeta + zeta_offset``. The
    matching opposing bunch is the one whose ``zeta`` is closest to
    ``zeta + zeta_offset`` within ``zeta_match_tol``; if none is found the
    particle receives no kick.
    """

    _xofields = {

        'scale_strength': xo.Float64,

        'zeta_offset': xo.Float64,
        'zeta_match_tol': xo.Float64,
        'zeta_period': xo.Float64,

        'other_beam_q0': xo.Float64,
        'other_beam_beta0': xo.Float64,
        'other_beam_gamma0': xo.Float64,

        'min_sigma_diff': xo.Float64,

        # Per-bunch description of the opposing beam
        'num_other_bunches': xo.Int64,
        'other_beam_zeta': xo.Float64[:],
        'other_beam_x': xo.Float64[:],
        'other_beam_y': xo.Float64[:],
        'other_beam_num_particles': xo.Float64[:],
        'other_beam_nemitt_x': xo.Float64[:],
        'other_beam_nemitt_y': xo.Float64[:],
        'other_beam_betx': xo.Float64[:],
        'other_beam_bety': xo.Float64[:],

    }

    _extra_c_sources = [
        '#include "xfields/beam_elements/beambeam_src/beambeam_multibunch_2d.h"',
    ]

    def __init__(self,
                    num_bunches=None,

                    scale_strength=1.,

                    zeta_offset=0.,
                    zeta_match_tol=1e-3,
                    zeta_period=0.,

                    other_beam_q0=0,
                    other_beam_beta0=1,
                    other_beam_gamma0=1,

                    other_particles=None,

                    other_beam_nemitt_x=0.,
                    other_beam_nemitt_y=0.,
                    other_beam_betx=1.,
                    other_beam_bety=1.,

                    min_sigma_diff=1e-10,

                    **kwargs):

        """
        Args:
            num_bunches (int): Maximum number of bunches of the opposing beam.
                Used to allocate the internal arrays. Inferred from
                ``other_particles`` if not given.
            scale_strength (float): Used to scale the beam-beam force strength.
                Scales ``other_beam_q0``.
            zeta_offset (float): A particle of this beam at ``zeta`` interacts
                with the opposing bunch located at ``zeta + zeta_offset``.
            zeta_match_tol (float): Maximum allowed distance in ``zeta`` between
                a particle's encounter position (``zeta + zeta_offset``) and the
                centroid of an opposing bunch for them to interact.
            zeta_period (float): Periodicity of the ``zeta`` bunch-label axis
                (e.g. ``n_slots * slot_spacing`` for a circular machine). If
                larger than zero, the encounter distance is evaluated modulo
                this period, so encounter offsets that wrap around the ring
                still find their partner. Zero (default) disables wrapping.
            other_beam_q0 (float): Charge sign of the opposing beam. -1 for
                electrons, +1 for protons or positrons.
            other_beam_beta0 (float): Relativistic beta of the opposing beam.
            other_beam_gamma0 (float): Relativistic gamma of the opposing beam.
                Used to convert the per-bunch normalized emittances into
                geometric ones.
            other_particles (xpart.Particles): Particles object of the opposing
                beam in which each active macroparticle represents one bunch.
                Its centroids (``x``, ``y``), longitudinal positions (``zeta``)
                and populations (``weight``) are loaded into the element (as by
                :meth:`update_from_other_beam`). Also used to infer
                ``num_bunches`` when not given explicitly.
            other_beam_nemitt_x, other_beam_nemitt_y (float or float array):
                Normalized transverse emittances of each opposing bunch. A
                scalar is broadcast to all bunches.
            other_beam_betx, other_beam_bety (float or float array): Beta
                functions of each opposing bunch at the interaction point. A
                scalar is broadcast to all bunches.
            min_sigma_diff (float): Round-beam kick (~2x faster) is used instead
                of the elliptical kick if
                ``fabs(sigma_x - sigma_y) < min_sigma_diff``.
        """

        if '_xobject' in kwargs.keys():
            self.xoinitialize(**kwargs)
            return

        # Determine the number of active bunches in the opposing beam
        n_active = 0
        if other_particles is not None:
            state = other_particles._context.nparray_from_context_array(
                other_particles.state)
            n_active = int((state > 0).sum())

        if num_bunches is None:
            num_bunches = n_active
        if num_bunches == 0:
            raise ValueError(
                'Specify `num_bunches` or `other_particles` to allocate the '
                'element.')
        assert num_bunches >= n_active, (
            '`num_bunches` must be >= the number of bunches in `other_particles`')

        self.xoinitialize(
            other_beam_zeta=num_bunches,
            other_beam_x=num_bunches,
            other_beam_y=num_bunches,
            other_beam_num_particles=num_bunches,
            other_beam_nemitt_x=num_bunches,
            other_beam_nemitt_y=num_bunches,
            other_beam_betx=num_bunches,
            other_beam_bety=num_bunches,
            **kwargs)

        self.scale_strength = scale_strength

        self.zeta_offset = zeta_offset
        self.zeta_match_tol = zeta_match_tol
        self.zeta_period = zeta_period

        self.other_beam_q0 = other_beam_q0
        self.other_beam_beta0 = other_beam_beta0
        self.other_beam_gamma0 = other_beam_gamma0

        self.min_sigma_diff = min_sigma_diff

        self.num_other_bunches = 0
        if other_particles is not None:
            self.update_from_other_beam(other_particles)

        # Per-bunch emittances / betas (scalars are broadcast to all bunches)
        self._set_per_bunch('other_beam_nemitt_x', other_beam_nemitt_x, num_bunches)
        self._set_per_bunch('other_beam_nemitt_y', other_beam_nemitt_y, num_bunches)
        self._set_per_bunch('other_beam_betx', other_beam_betx, num_bunches)
        self._set_per_bunch('other_beam_bety', other_beam_bety, num_bunches)

    def _set_per_bunch(self, name, value, num_bunches):
        value = np.atleast_1d(np.asarray(value, dtype=float))
        if value.size == 1:
            value = np.full(num_bunches, value[0])
        assert value.size <= num_bunches, (
            f'`{name}` has {value.size} entries but the element was allocated '
            f'for {num_bunches} bunches.')
        getattr(self, name)[:value.size] = self._arr2ctx(value)

    def update_from_other_beam(self, other_particles):

        """
        Load the centroid, longitudinal position and population of the bunches
        of the opposing beam from a :class:`xpart.Particles` object in which
        each (active) macroparticle represents one bunch.

        Should be called before tracking either beam through the beam-beam
        elements so that both kicks are computed from the bunch positions at the
        same turn (strong-strong simultaneity).

        Note: the per-bunch emittances and betas are static (set at construction
        time) and are not modified by this method.
        """

        ctx2np = self._buffer.context.nparray_from_context_array

        state = ctx2np(other_particles.state)
        mask = state > 0

        x = ctx2np(other_particles.x)[mask]
        y = ctx2np(other_particles.y)[mask]
        zeta = ctx2np(other_particles.zeta)[mask]
        weight = ctx2np(other_particles.weight)[mask]

        n = len(x)
        capacity = len(self.other_beam_zeta)
        if n > capacity:
            raise ValueError(
                f'The opposing beam has {n} bunches but the element was '
                f'allocated for {capacity}. Increase `num_bunches`.')

        # The tracking kernel finds the encounter partner by binary search, so
        # the bunches are stored sorted in zeta.
        order = np.argsort(zeta, kind='stable')

        self.num_other_bunches = n
        self.other_beam_zeta[:n] = self._arr2ctx(zeta[order])
        self.other_beam_x[:n] = self._arr2ctx(x[order])
        self.other_beam_y[:n] = self._arr2ctx(y[order])
        self.other_beam_num_particles[:n] = self._arr2ctx(weight[order])

    def get_sigma_x(self):
        """Per-bunch horizontal size sigma_x of the opposing bunches."""
        ctx2np = self._buffer.context.nparray_from_context_array
        n = self.num_other_bunches
        return np.sqrt(ctx2np(self.other_beam_betx)[:n]
                       * ctx2np(self.other_beam_nemitt_x)[:n]
                       / self.other_beam_gamma0)

    def get_sigma_y(self):
        """Per-bunch vertical size sigma_y of the opposing bunches."""
        ctx2np = self._buffer.context.nparray_from_context_array
        n = self.num_other_bunches
        return np.sqrt(ctx2np(self.other_beam_bety)[:n]
                       * ctx2np(self.other_beam_nemitt_y)[:n]
                       / self.other_beam_gamma0)
