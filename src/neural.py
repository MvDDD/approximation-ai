import json
import random
import math

class NeuralNet:
    def __init__(self, layers, act=lambda a: max(a, 0), out=math.tanh, random_gen=random.random):
        self.act = act
        self.out = out
        self.random = random_gen
        
        if isinstance(layers, dict) and 'nodes' in layers:
            self.nodes = layers['nodes']
            self.paths = layers['paths']
        else:
            self.nodes = [[0 for _ in range(i)] for i in layers]
            self.paths = []
            for layer in range(len(layers) - 1):
                path_layer = [
                    [start, end, 1 / (layers[layer] * layers[layer + 1])]
                    for start in range(layers[layer])
                    for end in range(layers[layer + 1])
                ]
                self.paths.append(path_layer)

    def mutate_paths(self, amount):
        if not hasattr(self, 'max'):
            self.paths = [
                [[start, end, weight + (self.random() - 0.5) * amount] for start, end, weight in layer]
                for layer in self.paths
            ]
        else:
            self.paths = [
                [[start, end, max(min(weight + (self.random() - 0.5) * amount, self.max), -self.max)]
                 for start, end, weight in layer]
                for layer in self.paths
            ]

    def mutate_nodes(self, amount):
        if not hasattr(self, 'max'):
            self.nodes = [
                [node + (self.random() - 0.5) * amount for node in layer]
                for layer in self.nodes
            ]
        else:
            self.nodes = [
                [max(min(node + (self.random() - 0.5) * amount, self.max), -self.max) for node in layer]
                for layer in self.nodes
            ]

    def clone(self):
        cloned_net = NeuralNet(
            {
                'nodes': [layer[:] for layer in self.nodes],
                'paths': [[path[:] for path in layer] for layer in self.paths],
                'act': self.act,
                'out': self.out
            }
        )
        if hasattr(self, 'max'):
            cloned_net.max = self.max
        return cloned_net

    def run(self, inputs):
        if len(inputs) != len(self.nodes[0]):
            raise TypeError(f"input size incorrect: {len(inputs)}")

        model = self.clone()
        model.nodes[0] = inputs[:]
        
        for layer_num, layer in enumerate(model.paths):
            for start, end, weight in layer:
                model.nodes[layer_num + 1][end] += model.nodes[layer_num][start] * weight
            model.nodes[layer_num + 1] = [self.act(a) for a in model.nodes[layer_num + 1]]

        return [self.out(a) for a in model.nodes[-1]]

    def __str__(self):
        return json.dumps({
            'nodes': [[round(v, 10) for v in layer] for layer in self.nodes],
            'paths': [[[start, end, round(weight, 10)] for start, end, weight in layer] for layer in self.paths],
            'act': self.act.__code__.co_code.decode("utf-8"),
            'out': self.out.__code__.co_code.decode("utf-8")
        })

    @staticmethod
    def from_string(data_str):
        data = json.loads(data_str)
        act = eval(f"lambda a: {data['act']}")
        out = eval(f"lambda a: {data['out']}")
        net = NeuralNet({'nodes': data['nodes'], 'paths': data['paths'], 'act': act, 'out': out})
        return net

    def from_net(self, net):
        return net.clone()

# Usage
if __name__ == "__main__":
    net = NeuralNet([3, 2, 1])
    print(net)