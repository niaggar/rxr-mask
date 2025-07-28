from rxrmask.core.formfactor import FormFactorModel

from dataclasses import dataclass


@dataclass
class Atom:
    Z: int
    name: str
    ff: FormFactorModel
    ffm: FormFactorModel | None = None
    mass: float | None = None  # in atomic mass units (g/mol)

    def __post_init__(self):
        if not isinstance(self.ff, FormFactorModel):
            raise TypeError("ff must be an instance of FormFactorModel")
        
        if self.mass is None:
            self.mass = self.load_atomic_mass()
    
    def load_atomic_mass(self):
        mass = None
        
        # TODO: Use relative path to the atomic mass file
        file = open("/Users/niaggar/Developer/mitacs/rxr-mask/rxrmask/materials/atomic_mass.txt", "r")
        lines = file.readlines()
        for line in lines:
            if line.split()[0] == self.name:
                mass = line.split()[1]
        file.close()

        if mass == None:
            raise NameError("Inputted formula not found in perovskite density database")

        return float(mass)


def get_atom(symbol: str, atoms: list[Atom]) -> Atom | None:
    for atom in atoms:
        if atom.name == symbol:
            return atom
    return None
