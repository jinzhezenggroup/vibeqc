"""Explicit chemists' integral and restricted-amplitude axis conventions."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MOBlock:
    """g[p,q,r,s]=(pq|rs); each tuple lists global spatial-MO columns.

    Slots, not a shorthand string, define the physical indices. A block named
    ``ovov`` for MP2 has slots (i,a,j,b), while t2 has axes (i,j,a,b).
    Arbitrary unique subsets and orderings are allowed within each slot.
    """

    slots: tuple[tuple[int, ...], ...]

    def __post_init__(self):
        object.__setattr__(self, "slots", tuple(tuple(x) for x in self.slots))
        if len(self.slots) != 4:
            raise ValueError("chemists' ERIs require four explicit MO slots")
        for slot in self.slots:
            if len(set(slot)) != len(slot) or any(
                type(i) is not int or i < 0 for i in slot
            ):
                raise ValueError("MO slots require unique nonnegative integer indices")

    @classmethod
    def from_spaces(cls, snapshot, spaces):
        """Expand a documented chemists'-slot label into explicit global MOs."""
        if len(spaces) != 4 or any(s not in "ov" for s in spaces):
            raise ValueError("use four o/v slots in chemists' order")
        indices = {
            "o": tuple(range(snapshot.nocc)),
            "v": tuple(range(snapshot.nocc, snapshot.nmo)),
        }
        return cls(tuple(indices[s] for s in spaces))

    @property
    def shape(self):
        return tuple(map(len, self.slots))

    def validate(self, snapshot):
        if any(i >= snapshot.nmo for slot in self.slots for i in slot):
            raise ValueError("MO index outside the reference snapshot")


def ovov_to_ijab(ovov):
    """Explicit adapter: chemists' (i a|j b) -> G[i,j,a,b], no antisymmetry."""
    value = np.asarray(ovov)
    if value.ndim != 4:
        raise ValueError("ovov must have four axes")
    return value.transpose(0, 2, 1, 3)
