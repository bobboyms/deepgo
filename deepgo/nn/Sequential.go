package nn

import (
	"tensors-processing/deepgo/linalg"
)

type Sequential struct {
	Layers []Layer
}

func NewSequential(layers []Layer) Sequential {
	return Sequential{
		Layers: layers,
	}
}

func (r *Sequential) Predict(input linalg.Matrix[float64]) linalg.Matrix[float64] {
	output := r.Layers[0].Forward(input)
	for _, dense := range r.Layers[1:] {
		output = dense.Forward(output)
	}
	return output
}
