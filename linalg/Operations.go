package linalg

import (
	"math"
)

func Log[T NumTypes](matrix Matrix[T]) Matrix[T] {

	row, col := matrix.LocalShape()
	data := make([]T, row*col)
	for i, t := range matrix.LocalData() {
		data[i] = T(math.Log(float64(t)))
	}
	return NewMatrix(data, row, col)
}

func Reduce[T NumTypes](matrix Matrix[T]) T {
	var sum T
	for _, t := range matrix.LocalData() {
		sum += t
	}
	return sum
}

func Mul[T NumTypes](matrixA, matrixB Matrix[T]) Matrix[T] {
	return ProcessOperation(matrixA, matrixB, MulVec[T])
}

func Div[T NumTypes](matrixA, matrixB Matrix[T]) Matrix[T] {
	return ProcessOperation(matrixA, matrixB, DivVec[T])
}

func Sub[T NumTypes](matrixA, matrixB Matrix[T]) Matrix[T] {
	return ProcessOperation(matrixA, matrixB, SubVec[T])
}

func Sum[T NumTypes](matrixA, matrixB Matrix[T]) Matrix[T] {
	return ProcessOperation(matrixA, matrixB, SumVec[T])
}
