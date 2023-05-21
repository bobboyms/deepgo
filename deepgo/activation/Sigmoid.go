package activation

import (
	"math"
	"tensors-processing/deepgo/linalg"
)

//def sigmoid_derivative(x):
//"""Derivada da função sigmóide"""
//return sigmoid(x) * (1 - sigmoid(x))

func SigmoidDerivative(matrix linalg.Matrix[float64]) linalg.Matrix[float64] {
	result := Sigmoid(matrix)
	return linalg.Mul(result, linalg.SubScalar(1, result))
}

func Sigmoid(matrix linalg.Matrix[float64]) linalg.Matrix[float64] {
	row, col := matrix.LocalShape()
	data := make([]float64, row*col)
	for i, x := range matrix.LocalData() {
		data[i] = 1 / (1 + math.Exp(-x))
	}
	return linalg.NewMatrix(data, row, col)
}
