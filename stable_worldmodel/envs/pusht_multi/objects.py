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
