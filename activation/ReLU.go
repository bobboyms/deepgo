package activation

import (
	"tensors-processing/linalg"
	"tensors-processing/nn"
)

type ReLUStruct struct {
}

func NewReLU() nn.Layer {
	return &ReLUStruct{}
}

func (r ReLUStruct) Forward(inputs linalg.Matrix[float64]) linalg.Matrix[float64] {
	return ReLU(inputs)
}

func ReLU(matrix linalg.Matrix[float64]) linalg.Matrix[float64] {

	relu := func(value float64) float64 {
		if value > 0 {
			return value
		} else {
			return 0
		}
	}

	row, col := matrix.LocalShape()
	data := make([]float64, row*col)

	for i, t := range matrix.LocalData() {
		data[i] = relu(t)
	}

	return linalg.NewMatrix(data, row, col)
}
