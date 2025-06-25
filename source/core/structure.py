from .layer import Layer


class Structure:
    def __init__(self, name:str):
        self.name = name
        self.layers:list[Layer] = []

    def add_compound(self, layer: Layer):
        self.layers.append(layer)

    def add(self, layer:Layer, repeat:int=1):
        for id in range(repeat):
            layer.id = layer.name + str(id)
            newLayer = Layer(layer)
            self.layers.append(newLayer)
