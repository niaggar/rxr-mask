from dataclasses import dataclass, field

from source.core.compound import Compound

@dataclass
class Structure:
    name: str
    compounds: list[Compound] = field(default_factory=list)

    def add(self, compound: Compound, repeat: int = 1):
        if not isinstance(compound, Compound):
            raise TypeError("Expected a Compound instance.")

        for _ in range(repeat):
            self.compounds.append(compound)
