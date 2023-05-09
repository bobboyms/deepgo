package nn

import "tensors-processing/linalg"

type Layer interface {
	Forward(inputs linalg.Matrix[float64]) linalg.Matrix[float64]
}
