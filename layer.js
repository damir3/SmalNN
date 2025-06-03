import { Neuron } from './neuron.js'

class Layer {
    constructor(dimIn, dimOut, activationFunction = 'relu') {
        this.neurons = Array.from({ length: dimOut }, () => new Neuron(dimIn, activationFunction))
    }

    parameters() {
        return this.neurons.flatMap((neuron) => neuron.parameters())
    }

    forward(inputs) {
        return this.neurons.map((neuron) => neuron.forward(inputs))
    }
}

export { Layer }
