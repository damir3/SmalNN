
const BACKWARD_FUNC = {
    '+': function(value) {
        const [type, parent1, parent2] = value._op
        parent1.grad += value.grad
        parent2.grad += value.grad
    },
    '*': function(value) {
        const [type, parent1, parent2] = value._op
        parent1.grad += value.grad * parent2.data
        parent2.grad += value.grad * parent1.data
    },
    '^': function(value) {
        const [type, parent, power] = value._op
        parent.grad += value.grad * power * Math.pow(parent.data, power - 1)
    },
    'exp': function(value) {
        const [type, parent] = value._op
        parent.grad += value.grad * value.data
    },
    'log': function(value) {
        const [type, parent] = value._op
        if (parent.data <= 0) {
            throw new Error(`Logarithm of non-positive number ${parent.data}`)
        }
        parent.grad += value.grad / parent.data
    },
    'tanh': function(value) {
        const [type, parent] = value._op
        parent.grad += value.grad * (1 - Math.pow(value.data, 2))
    },
    'sigmoid': function(value) {
        const [type, parent] = value._op
        parent.grad += value.grad * value.data * (1 - value.data)
    },
    'relu': function(value) {
        const [type, parent] = value._op
        if (parent.data > 0) {
            parent.grad += value.grad
        }
    },
    'gelu': function(value) {
        const [type, parent] = value._op
        const x = parent.data, x3 = x ** 3
        const axbx3 = 0.797885 * x + 0.0356774 * x3
        const dGELU = 0.5 + 0.5 * Math.tanh(axbx3) + (0.398942 * x + 0.0535161 * x3) / (Math.cosh(axbx3) ** 2)
        parent.grad += value.grad * dGELU
    },
}

class Value {
    constructor(value, op = null) {
        this.data = value
        this.grad = 0
        this._op = op
    }

    add(other) {
        if (!(other instanceof Value)) {
            other = new Value(other)
        }
        return new Value(this.data + other.data, ['+', this, other])
    }

    mul(other) {
        if (!(other instanceof Value)) {
            other = new Value(other)
        }
        return new Value(this.data * other.data, ['*', this, other])
    }

    sub(other) {
        const negOther = other instanceof Value ? other.mul(-1) : new Value(-other)
        return this.add(negOther)
    }

    pow(value) {
        if (typeof value !== 'number') {
            throw new Error(`Power "${value}" must be a number`)
        }
        return new Value(Math.pow(this.data, value), ['^', this, value])
    }

    div(other) {
        if (other instanceof Value) {
            return this.mul(other.pow(-1))
        }
        return this.mul(1 / other)
    }

    exp() {
        return new Value(Math.exp(this.data), ['exp', this])
    }

    log() {
        if (this.data <= 0) {
            throw new Error(`Logarithm of non-positive number ${this.data}`)
        }
        return new Value(Math.log(this.data), ['log', this])
    }

    tanh() {
        return new Value(Math.tanh(this.data), ['tanh', this])
    }

    sigmoid() {
        return new Value(1 / (1 + Math.exp(-this.data)), ['sigmoid', this])
    }

    relu() {
        return new Value(Math.max(0, this.data), ['relu', this])
    }

    gelu() {
        return new Value(0.5 * this.data * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (this.data + 0.044715 * (this.data ** 3)))), ['gelu', this])
    }

    backward() {
        const orderedValues = [] // ordered list of values from inputs to outputs
        const valueSet = new Set()
        this.traverse((v) => {
            if (!valueSet.has(v)) {
                valueSet.add(v)
                orderedValues.push(v)
            }
        })

        orderedValues.reverse() // reverse the order for backpropagation (from outputs to inputs)

        orderedValues.forEach((v) => { v.grad = 0 }) // reset gradients

        this.grad = 1
        orderedValues.forEach((v) => {
            if (v._op != null) {
                BACKWARD_FUNC[v._op[0]](v)
            }
        })
    }

    traverse(callback) {
        this._op?.forEach(function(parent) {
            if (parent instanceof Value) {
                parent.traverse(callback)
            }
        })
        callback(this)
    }

    toString() {
        return `Data(${this.data}) Grad(${this.grad})`
    }
}

export { Value }
