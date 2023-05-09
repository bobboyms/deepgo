package nn

import (
	"tensors-processing/linalg"
	"tensors-processing/nn/layer"
)

type RNN struct {
	Layers []layer.Dense
}

func NewRNN(layers []layer.Dense) RNN {
	return RNN{
		Layers: layers,
	}
}

func (r *RNN) Predict(input linalg.Matrix[float64]) linalg.Matrix[float64] {
	output := r.Layers[0].Forward(input)
	for _, dense := range r.Layers[1:] {
		output = dense.Forward(output)
	}
	return output
}
