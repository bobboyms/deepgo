package nn

import (
	"tensors-processing/linalg"
)

type Dense struct {
	Weights linalg.Matrix[float64]
	Biases  linalg.Matrix[float64]
	//Activation func(matrix linalg.Matrix[float64]) linalg.Matrix[float64]
}

// activation func(matrix linalg.Matrix[float64]) linalg.Matrix[float64]
func NewDense(numInputs, numNeurons int) Layer {

	w := linalg.CreateRandomData(numInputs * numNeurons)
	b := linalg.CreateRandomData(numNeurons)

	return &Dense{
		Weights: linalg.NewMatrix(w, numInputs, numNeurons),
		Biases:  linalg.NewMatrix(b, 1, numNeurons),
	}
}

func (d *Dense) Forward(inputs linalg.Matrix[float64]) linalg.Matrix[float64] {
	dotResult := linalg.Dot(inputs, d.Weights)
	return linalg.Sum(dotResult, d.Biases)
}
