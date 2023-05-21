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

func SubScalar[T NumTypes](value T, matrix Matrix[T]) Matrix[T] {
	data := matrix.LocalData()
	temp := make([]T, len(data))
	for i, t := range data {
		temp[i] = value - t
	}
	row, col := matrix.LocalShape()
	return NewMatrix(temp, row, col)
}

func DivScalar[T NumTypes](matrix Matrix[T], value T) Matrix[T] {
	data := matrix.LocalData()
	temp := make([]T, len(data))
	for i, t := range data {
		temp[i] = t / value
	}
	row, col := matrix.LocalShape()
	return NewMatrix(temp, row, col)
}

func MulScalar[T NumTypes](value T, matrix Matrix[T]) Matrix[T] {
	data := matrix.LocalData()
	temp := make([]T, len(data))
	for i, t := range data {
		temp[i] = t * value
	}
	row, col := matrix.LocalShape()
	return NewMatrix(temp, row, col)
}

func SumAxis[T NumTypes](matrix Matrix[T], axis int) Matrix[T] {
	sRow, sCol := matrix.LocalShape()
	rows := GetRow(matrix.LocalData(), sRow, sCol)
	if axis == 0 {
		size := len(rows[0])
		sum := make([]T, size)
		for _, row := range rows {
			for i, val := range row {
				sum[i] += val
			}
		}
		return NewMatrix(sum, 1, size)
	} else if axis == 1 {
		size := len(rows[0])
		sum := make([]T, size)
		for i, row := range rows {
			for _, val := range row {
				sum[i] += val
			}
		}
		return NewMatrix(sum, 1, size)
	}

	panic("invalid axis, it should be either 0 or 1")
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
