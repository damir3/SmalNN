import { Layer } from './layer.js'

class SmallNN {
    constructor(inDim, layerDims, activationFunction = 'relu',) {
        const sizes = [inDim, ...layerDims]
        this.layers = Array.from(layerDims, (outDim, i) => new Layer(sizes[i], outDim, activationFunction))
    }

    parameters() {
        return this.layers.flatMap(layer => layer.parameters())
    }

    forward(x, batch = false) {
        if (batch) {
            return x.map(x => this.forward(x))
        }

        this.layers.forEach((layer) => { x = layer.forward(x) })
        return x
    }

    static _extractInputs(inputs) {
        return inputs.map(i => Array.isArray(i) ? i[0] : i)
    }

    static MSELoss(inputs, targets) {
        const size = targets.length
        if (inputs.length !== size) {
            throw new Error("Inputs and targets must have the same length")
        }

        inputs = SmallNN._extractInputs(inputs)

        let loss
        targets.forEach((target, i) => {
            const c = inputs[i].sub(target).pow(2)
            loss = loss ? loss.add(c) : c
        })
        return size > 1 ? loss.div(size) : loss
    }

    static SoftMax(inputs) {
        inputs = SmallNN._extractInputs(inputs)

        const expInputs = inputs.map(i => i.exp())
        const sumExp = expInputs.reduce((acc, expI) => acc ? acc.add(expI) : expI)
        const invSumExp = sumExp.pow(-1)
        return expInputs.map(expI => expI.mul(invSumExp))
    }

    static CrossEntropyLoss(inputs, targets) {
        const size = targets.length
        if (inputs.length !== size) {
            throw new Error("Inputs and targets must have the same length")
        }

        const softMax = SmallNN.SoftMax(inputs)

        let loss
        targets.forEach((target, i) => {
            const c = softMax[i].log().mul(target)
            loss = loss ? loss.add(c) : c
        })
        return loss.mul(-1)
    }
}

export { SmallNN }