package activation

import (
	"math"
	"tensors-processing/deepgo/linalg"
)

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
