package loss

import (
	"math"
	"tensors-processing/deepgo/linalg"
)

//def mse_loss_derivative(y_true, y_pred):
//"""Derivada da função de custo - Erro quadrático médio"""
//return 2 * (y_pred - y_true) / y_true.size

func MeanSquaredErrorDerivative(yTrue, yPred linalg.Matrix[float64]) linalg.Matrix[float64] {
	yRow, yCol := yTrue.LocalShape()
	multi := linalg.MulScalar(2, linalg.Sub(yPred, yTrue))
	return linalg.DivScalar(multi, float64(yRow*yCol))
}

func Mse(yTrue, yPred linalg.Matrix[float64]) float64 {
	//rows := len(yTrue)
	//cols := len(yTrue[0])

	rows, cols := yTrue.LocalShape()
	yTdata := linalg.GetRow(yTrue.LocalData(), rows, cols)
	yPdata := linalg.GetRow(yPred.LocalData(), rows, cols)

	sum := 0.0
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			diff := yTdata[i][j] - yPdata[i][j]
			sum += math.Pow(diff, 2)
		}
	}

	mean := sum / float64(rows*cols)
	return mean
}
