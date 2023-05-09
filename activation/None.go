package activation

import "tensors-processing/linalg"

func None(matrix linalg.Matrix[float64]) linalg.Matrix[float64] {
	return matrix
}
