from dataclasses import dataclass
import numpy as np
from .atom import Atom
import pint

from .layer import Layer


@dataclass
class Compound:
    id: str
    name: str
    thickness: float          # en u.nm
    density: float            # en u.kg/u.m**3
    atoms: list[Atom] | None = None
    layers: list[Layer] | None = None

    def create_layer(self, n_layers:int=1):
        layers = []
        for _ in range(n_layers):
            layers.append(Layer())

