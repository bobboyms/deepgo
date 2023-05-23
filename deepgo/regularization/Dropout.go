package regularization

import (
	"math/rand"
	"tensors-processing/deepgo/linalg"
)

func Dropout(input linalg.Matrix[float64], dropoutRate float64) linalg.Matrix[float64] {

	row, col := input.LocalShape()
	rows := linalg.GetRow(input.LocalData(), row, col)

	output := make([][]float64, row)
	for i := 0; i < row; i++ {
		output[i] = make([]float64, col)
		for j := 0; j < len(rows[i]); j++ {
			if rand.Float64() >= dropoutRate {
				output[i][j] = rows[i][j] / (1.0 - dropoutRate)
			}
		}
	}
	return linalg.NewMatrixFrom2D(output, row, col)
}
