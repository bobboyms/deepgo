package test

import (
	"tensors-processing/deepgo/activation"
	"tensors-processing/deepgo/linalg"
	"tensors-processing/deepgo/nn"
	"testing"
)

func TestForward(t *testing.T) {
	inputs := linalg.NewMatrix([]float64{1, 2, 3}, 1, 3)
	l1 := nn.NewDense(3, 2, activation.Sigmoid)
	l1.Forward(inputs).Print()
}
