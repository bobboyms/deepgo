package test

import (
	"tensors-processing/deepgo/linalg"
	"tensors-processing/deepgo/loss"
	"testing"
)

func TestMeanSquaredError(t *testing.T) {

	yTrue := linalg.NewMatrix([]float64{0.0}, 1, 1)
	yPred := linalg.NewMatrix([]float64{0.67210983}, 1, 1)

	got := loss.Mse(yTrue, yPred)
	want := 0.4517316235826289

	if got != want {
		t.Errorf("got %f, wanted %f", got, want)
	}

}

func TestMeanSquaredErrorDerivative(t *testing.T) {
	yTrue := linalg.NewMatrix([]float64{0, 1, 0}, 1, 3)
	yPred := linalg.NewMatrix([]float64{1, 1, 1}, 1, 3)

	got := loss.MeanSquaredErrorDerivative(yTrue, yPred)
	//[0.66666667, 0.        , 0.66666667]
	if got.LocalData()[0] != 0.6666666666666666 {
		t.Errorf("got %f, wanted %f", got.LocalData()[0], 0.6666666666666666)
	}
	if got.LocalData()[1] != 0 {
		t.Errorf("got %f, wanted %f", got.LocalData()[1], 0.0)
	}
	if got.LocalData()[2] != 0.6666666666666666 {
		t.Errorf("got %f, wanted %f", got.LocalData()[2], 0.6666666666666666)
	}
}
