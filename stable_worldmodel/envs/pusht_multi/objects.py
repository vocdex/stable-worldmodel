"""Canonical object library for the multi-object PushT environment.

Identity↔label is fixed across all scene compositions so that per-slot
analyses are comparable across {A,B}, {B,C}, {A,B,C}, etc. Appearance
(color, scale) may still be perturbed per episode via the variation
space, but the segmentation label is a property of the identity.

Labels:
    LABEL_BG    = 0  — background pixels (no body)
    LABEL_AGENT = 1  — the kinematic agent (circle)
    LABEL_*     >=2  — one per named object identity
"""

from __future__ import annotations

from dataclasses import dataclass


LABEL_BG = 0
LABEL_AGENT = 1


@dataclass(frozen=True)
class ObjectSpec:
    """Static definition of one named object identity.

    Attributes:
        shape: Identifier consumed by the PushTMulti shape dispatcher.
            One of 'T', 'L', 'Z', 'square', 'I', 'small_tee', '+', 'o'.
        color: Default pygame color name.
        scale: Default scale (consumed by the shape constructors).
        mass: pymunk body mass.
        friction: pymunk shape friction.
        has_orientation: If False, the object's angle is ignored in the
            success criterion (e.g., a circle has no meaningful angle).
        label: Segmentation label. Must be unique and >= 2.
    """

    shape: str
    color: str
    scale: float
    mass: float
    friction: float
    has_orientation: bool
    label: int

    @property
    def bounding_radius(self) -> float:
        """Rotation-invariant circumscribed-circle radius (pixels).

        Worst-case distance from the body's local origin to any point on
        the shape, computed from the polygon vertex sets defined in
        `pusht_multi.env._add_*`. Used by the start/goal-position
        rejection sampler to enforce non-overlap independent of angle.
        """
        s = self.scale
        if self.shape == 'T':
            # stem extends to y = 4*s with x = ±s/2
            return float(((s / 2) ** 2 + (4 * s) ** 2) ** 0.5)
        if self.shape == 'small_tee':
            return float(((s / 2) ** 2 + (2 * s) ** 2) ** 0.5)
        if self.shape == 'L':
            # 60 x 30 rectangles at scale=30 → diag at far corner
            return float(((2 * s) ** 2 + s ** 2) ** 0.5)
        if self.shape == 'Z':
            return float(((s) ** 2 + (s / 2) ** 2) ** 0.5) * 2  # ~conservative
        if self.shape == 'square':
            return float((2 * (s ** 2)) ** 0.5)
        if self.shape == 'I':
            return float(((s / 2) ** 2 + (2 * s) ** 2) ** 0.5)
        if self.shape == '+':
            return float(((3 * s / 2) ** 2 + (s / 2) ** 2) ** 0.5)
        if self.shape == 'o':
            return float(0.375 * s)
        raise ValueError(f'Unknown shape {self.shape!r}')


# Effective bounding radius of the kinematic agent (always a circle).
# Mirrors `_add_kinematic_circle`: radius = 0.375 * scale, agent default
# scale = 40 → 15 px. Used by the overlap sampler.
def agent_bounding_radius(agent_scale: float) -> float:
    return float(0.375 * agent_scale)


OBJECT_LIBRARY: dict[str, ObjectSpec] = {
    'A': ObjectSpec(shape='T',         color='LightSlateGray', scale=30, mass=1.0, friction=1.0, has_orientation=True,  label=2),
    'B': ObjectSpec(shape='I',         color='Orange',         scale=30, mass=1.0, friction=1.0, has_orientation=True,  label=3),
    'C': ObjectSpec(shape='o',         color='SeaGreen',       scale=40, mass=0.5, friction=0.3, has_orientation=False, label=4),
    'D': ObjectSpec(shape='square',    color='Purple',         scale=30, mass=2.0, friction=1.5, has_orientation=True,  label=5),
    'E': ObjectSpec(shape='+',         color='Crimson',        scale=30, mass=1.0, friction=1.0, has_orientation=True,  label=6),
    # F intentionally avoids RoyalBlue — that's the agent's default color,
    # and visual overlap would make it impossible to tell pusher from F by
    # color alone (segmentation labels are still distinct: agent=1, F=7).
    'F': ObjectSpec(shape='L',         color='Gold',           scale=30, mass=1.0, friction=1.0, has_orientation=True,  label=7),
}


ALL_OBJECTS: tuple[str, ...] = tuple(OBJECT_LIBRARY.keys())
