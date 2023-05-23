package test

import (
	"math/rand"
	"tensors-processing/deepgo/linalg"
	"tensors-processing/deepgo/regularization"
	"testing"
)

func TestDropout(t *testing.T) {
	X := [][]float64{
		{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
		{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
		{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
		{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
		{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
	}

	dropoutRate := 0.8

	rand.Seed(0)

	result := regularization.Dropout(linalg.NewMatrixFrom2D(X, 5, 17), dropoutRate)

	result.Print()

	//// Verificar se as dimensões de result são iguais às dimensões de X
	//if len(result) != len(X) {
	//	t.Errorf("A matriz result tem número incorreto de linhas. Esperado %d, recebido %d", len(X), len(result))
	//}
	//
	//for i := 0; i < len(X); i++ {
	//	if len(result[i]) != len(X[i]) {
	//		t.Errorf("A matriz result tem número incorreto de colunas na linha %d. Esperado %d, recebido %d", i, len(X[i]), len(result[i]))
	//	}
	//}
	//
	//// Verificar se a multiplicação foi aplicada corretamente
	//for i := 0; i < len(X); i++ {
	//	for j := 0; j < len(X[i]); j++ {
	//		if result[i][j] != 0 && result[i][j] != X[i][j] {
	//			t.Errorf("Valor incorreto na posição (%d, %d). Esperado %f ou %f, recebido %f", i, j, 0.0, X[i][j]*(1.0/(1.0-dropoutRate)), result[i][j])
	//		}
	//	}
	//}
}
