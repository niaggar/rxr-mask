from dataclasses import dataclass, field

from source.core.compound import Compound

@dataclass
class Structure:
    name: str
    compounds: list[Compound | None] = field(default_factory=list)
    n_compounds: int = 0
    n_layers: int = 0

    def __init__(self, name: str, n_compounds: int):
        self.name = name
        self.n_compounds = n_compounds
        self.compounds = [None] * n_compounds

    def add_compound(self, index: int, compound: Compound):
        if index < 0 or index >= self.n_compounds:
            raise IndexError("Index out of range for compounds.")
        if not isinstance(compound, Compound):
            raise TypeError("Expected a Compound instance.")

        self.compounds[index] = compound
        self.n_layers += compound.n_layers

    def __repr__(self) -> str:
        return f"Structure(name={self.name}, n_compounds={self.n_compounds}, n_layers={self.n_layers})"

