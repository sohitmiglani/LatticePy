class amino_acid():
    def __init__(self, polarity, coordinates, polymer_id=None):
        self.polarity = polarity
        self.coordinates = coordinates
        self.next = None
        self.previous = None
        self.polymer = polymer_id

class linker_bead():
    def __init__(self, polarity, coordinates, neighbor=None, polymer_id=None):
        self.polarity = polarity
        self.coordinates = coordinates
        self.neighbor = neighbor
        self.polymer = polymer_id