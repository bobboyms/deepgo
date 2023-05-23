package nn

import (
	linalg "tensors-processing/deepgo/linalg"
)

type Dense struct {
	Weights    linalg.Matrix[float64]
	Biases     linalg.Matrix[float64]
	Activation func(matrix linalg.Matrix[float64]) linalg.Matrix[float64]
}

func NewDense(numInputs, numNeurons int, activation func(matrix linalg.Matrix[float64]) linalg.Matrix[float64]) Layer {

	return &Dense{
		Activation: activation,
		Weights:    linalg.NormalDistribution(numInputs, numNeurons),
		Biases:     linalg.NormalDistribution(1, numNeurons),
	}
}

func (d *Dense) B() linalg.Matrix[float64] {
	return d.Biases
}

func (d *Dense) ChangeB(b linalg.Matrix[float64]) {
	d.Biases = b
}

func (d *Dense) ChangeW(w linalg.Matrix[float64]) {
	d.Weights = w
}

func (d *Dense) W() linalg.Matrix[float64] {
	return d.Weights
}

func (d *Dense) Forward(inputs linalg.Matrix[float64]) linalg.Matrix[float64] {
	dotResult := linalg.Dot(inputs, d.Weights)
	return d.Activation(linalg.Sum(dotResult, d.Biases))
}
