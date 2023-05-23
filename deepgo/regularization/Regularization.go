package regularization

import "tensors-processing/deepgo/linalg"

type Regularization interface {
	Apply(input linalg.Matrix[float64]) linalg.Matrix[float64]
}
