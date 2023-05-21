package nn

import (
	"tensors-processing/deepgo/linalg"
)

type Layer interface {
	Forward(inputs linalg.Matrix[float64]) linalg.Matrix[float64]
	W() linalg.Matrix[float64]
	B() linalg.Matrix[float64]
	ChangeW(w linalg.Matrix[float64])
	ChangeB(b linalg.Matrix[float64])
}
