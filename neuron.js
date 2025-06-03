import { Value } from './value.js'

class Neuron {
    constructor(dim, activationFunction = 'relu') {
        this.weights = Array.from({ length: dim }, () => new Value(Math.random() * 2 - 1))
        this.bias = new Value(0)
        this.actFunc = activationFunction
    }

    parameters() {
        return [...this.weights, this.bias]
    }

    forward(inputs) {
        if (inputs.length !== this.weights.length) {
            throw new Error("Input length must match weights length")
        }

        let sum
        inputs.forEach((input, i) => {
            sum = (sum ?? this.bias).add(this.weights[i].mul(input))
        })

        return this.actFunc ? sum[this.actFunc]() : sum
    }
}

export { Neuron }
