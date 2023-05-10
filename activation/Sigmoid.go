package activation

import (
	"math"
	"tensors-processing/linalg"
	"tensors-processing/nn"
)

type SigmoidStruct struct {
}

func NewSigmoid() nn.Layer {
	return &SigmoidStruct{}
}

func (s SigmoidStruct) Forward(inputs linalg.Matrix[float64]) linalg.Matrix[float64] {
	return Sigmoid(inputs)
}

func Sigmoid(matrix linalg.Matrix[float64]) linalg.Matrix[float64] {
	row, col := matrix.LocalShape()
	data := make([]float64, row*col)
	for i, x := range matrix.LocalData() {
		data[i] = 1 / (1 + math.Exp(-x))
	}
	return linalg.NewMatrix(data, row, col)
}
