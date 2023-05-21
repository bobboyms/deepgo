package test

import (
	"math"
	"tensors-processing/deepgo/linalg"
	"testing"
)

func TestNormalDistribution(t *testing.T) {

	sample := linalg.NormalDistribution(2, 3)

	// Verificar se os elementos da amostra estão dentro do intervalo esperado (3 desvios padrão)
	for _, value := range sample.LocalData() {
		if math.Abs(value) > 3 {
			t.Errorf("O valor da amostra está fora do intervalo esperado. Valor: %.4f", value)
		}
	}

}

func TestSumAxis(t *testing.T) {
	input := linalg.NewMatrix([]float64{1, 2, 3, 2, 0, 3, 6, 2, 3}, 3, 3)
	got := linalg.SumAxis(input, 0)

	if got.LocalData()[0] != 9.0 {
		t.Errorf("got %f, wanted %f", got.LocalData()[0], 9.0)
	}
	if got.LocalData()[1] != 4.0 {
		t.Errorf("got %f, wanted %f", got.LocalData()[1], 4.0)
	}
	if got.LocalData()[2] != 9.0 {
		t.Errorf("got %f, wanted %f", got.LocalData()[2], 9.0)
	}

	got = linalg.SumAxis(input, 1)

	if got.LocalData()[0] != 6.0 {
		t.Errorf("got %f, wanted %f", got.LocalData()[0], 6.0)
	}
	if got.LocalData()[1] != 5.0 {
		t.Errorf("got %f, wanted %f", got.LocalData()[1], 5.0)
	}
	if got.LocalData()[2] != 11.0 {
		t.Errorf("got %f, wanted %f", got.LocalData()[2], 11.0)
	}

	//6 5 11

}
