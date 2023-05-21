package test

import (
	"tensors-processing/deepgo/activation"
	"tensors-processing/deepgo/linalg"
	"testing"
)

func TestSigmoidDerivative(t *testing.T) {
	got := activation.SigmoidDerivative(linalg.NewMatrix([]float64{0, 1, -0.2}, 1, 3))
	if got.LocalData()[0] != 0.25 {
		t.Errorf("got %f, wanted %f", got.LocalData()[0], 0.25)
	}
	if got.LocalData()[1] != 0.19661193324148185 {
		t.Errorf("got %f, wanted %f", got.LocalData()[1], 0.19661193324148185)
	}
	if got.LocalData()[2] != 0.24751657271185995 {
		t.Errorf("got %f, wanted %f", got.LocalData()[2], 0.24751657271185995)
	}

	activation.SigmoidDerivative(linalg.NewMatrix([]float64{0.63891177}, 1, 1)).Print()

}

func TestSigmoid(t *testing.T) {
	got := activation.Sigmoid(linalg.NewMatrix([]float64{0, 1, -0.2}, 1, 3))
	if got.LocalData()[0] != 0.5 {
		t.Errorf("got %f, wanted %f", got.LocalData()[0], 0.5)
	}
	if got.LocalData()[1] != 0.7310585786300049 {
		t.Errorf("got %f, wanted %f", got.LocalData()[1], 0.7310585786300049)
	}
	if got.LocalData()[2] != 0.45016600268752216 {
		t.Errorf("got %f, wanted %f", got.LocalData()[2], 0.45016600268752216)
	}
}
