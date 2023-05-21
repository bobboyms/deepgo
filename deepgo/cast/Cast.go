package cast

import (
	"tensors-processing/deepgo/linalg"
)

func Float64ToT[T linalg.NumTypes](matrix linalg.Matrix[float64]) linalg.Matrix[T] {
	row, col := matrix.LocalShape()

	data := make([]T, row*col)

	for i, t := range matrix.LocalData() {
		data[i] = T(t)
	}

	return linalg.NewMatrix(data, row, col)
}

func CastToInt[T linalg.NumTypes](matrix linalg.Matrix[T]) linalg.Matrix[int] {
	row, col := matrix.LocalShape()
	return linalg.NewMatrix(ToInt(matrix.LocalData()), row, col)
}

func CastToInt8[T linalg.NumTypes](matrix linalg.Matrix[T]) linalg.Matrix[int8] {
	row, col := matrix.LocalShape()
	return linalg.NewMatrix(ToInt8(matrix.LocalData()), row, col)
}

func CastToInt16[T linalg.NumTypes](matrix linalg.Matrix[T]) linalg.Matrix[int16] {
	row, col := matrix.LocalShape()
	return linalg.NewMatrix(ToInt16(matrix.LocalData()), row, col)
}

func CastToInt32[T linalg.NumTypes](matrix linalg.Matrix[T]) linalg.Matrix[int32] {
	row, col := matrix.LocalShape()
	return linalg.NewMatrix(ToInt32(matrix.LocalData()), row, col)
}

func CastToInt64[T linalg.NumTypes](matrix linalg.Matrix[T]) linalg.Matrix[int64] {
	row, col := matrix.LocalShape()
	return linalg.NewMatrix(ToInt64(matrix.LocalData()), row, col)
}

func CastToFloat32[T linalg.NumTypes](matrix linalg.Matrix[T]) linalg.Matrix[float32] {
	row, col := matrix.LocalShape()
	return linalg.NewMatrix(ToFloat32(matrix.LocalData()), row, col)
}

func CastToFloat64[T linalg.NumTypes](matrix linalg.Matrix[T]) linalg.Matrix[float64] {
	row, col := matrix.LocalShape()
	return linalg.NewMatrix(ToFloat64(matrix.LocalData()), row, col)
}

func ToInt[T linalg.NumTypes](data []T) []int {

	newData := make([]int, len(data))

	for i, val := range data {
		newData[i] = int(val)
	}

	return newData
}

func ToInt8[T linalg.NumTypes](data []T) []int8 {

	newData := make([]int8, len(data))

	for i, val := range data {
		newData[i] = int8(val)
	}

	return newData
}

func ToInt16[T linalg.NumTypes](data []T) []int16 {

	newData := make([]int16, len(data))

	for i, val := range data {
		newData[i] = int16(val)
	}

	return newData
}

func ToInt32[T linalg.NumTypes](data []T) []int32 {

	newData := make([]int32, len(data))

	for i, val := range data {
		newData[i] = int32(val)
	}

	return newData
}

func ToInt64[T linalg.NumTypes](data []T) []int64 {

	newData := make([]int64, len(data))

	for i, val := range data {
		newData[i] = int64(val)
	}

	return newData
}

func ToFloat32[T linalg.NumTypes](data []T) []float32 {
	newData := make([]float32, len(data))

	for i, val := range data {
		newData[i] = float32(val)
	}

	return newData
}

func ToFloat64[T linalg.NumTypes](data []T) []float64 {
	newData := make([]float64, len(data))

	for i, val := range data {
		newData[i] = float64(val)
	}

	return newData
}
