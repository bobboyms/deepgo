package activation

import (
	"tensors-processing/deepgo/linalg"
)

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
