package loss

import (
	"tensors-processing/linalg"
)

func CrossEntropy(yTrue, yPred linalg.Matrix[float64]) float64 {
	return -linalg.Reduce(linalg.Mul(yTrue, linalg.Log(yPred)))
}
