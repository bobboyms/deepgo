package activation

import (
	"math"
	"tensors-processing/linalg"
	"tensors-processing/nn"
)

type SoftmaxStruct struct {
}

func NewSoftmax() nn.Layer {
	return &SoftmaxStruct{}
}

func (s *SoftmaxStruct) Forward(inputs linalg.Matrix[float64]) linalg.Matrix[float64] {
	return Softmax(inputs)
}

func Softmax(matrix linalg.Matrix[float64]) linalg.Matrix[float64] {

	row, col := matrix.LocalShape()
	data := make([]float64, row*col)
	sum := 0.0
	for i, v := range matrix.LocalData() {
		data[i] = math.Exp(v)
		sum += data[i]
	}

	for i := range data {
		data[i] /= sum
	}

	return linalg.NewMatrix(data, row, col)
}
