package linalg

import (
	"fmt"
	"math/rand"
	"time"
)

type Matrix[T NumTypes] interface {
	Size() int
	LocalShape() (int, int)
	LocalData() []T
	Print()
	Transpose() Matrix[T]
	Clone() Matrix[T]
}

type MatrixStruct[T NumTypes] struct {
	Data []T
	Row  int
	Col  int
}

type NumTypes interface {
	int | int8 | int16 | int32 | int64 | float64 | float32
}

func NewMatrix[T NumTypes](data []T, row, col int) Matrix[T] {
	ValidateShapeFromData(data, row, col)
	return &MatrixStruct[T]{
		Data: data,
		Row:  row,
		Col:  col,
	}
}

func NewBroadcasting[T NumTypes](matrix Matrix[T], row int) Matrix[T] {

	_, col := matrix.LocalShape()
	matrixData := matrix.LocalData()

	temData := make([][]T, row)
	for i := 0; i < row; i++ {
		temData[i] = matrixData
	}

	return NewMatrixFrom2D(temData, row, col)
}

func NewMatrixFrom2D[T NumTypes](data [][]T, row, col int) Matrix[T] {
	return NewMatrix(FlatData(data), row, col)
}

func FlatData[T NumTypes](data [][]T) []T {

	var flated []T
	for _, datum := range data {
		for _, v := range datum {
			flated = append(flated, v)
		}
	}
	return flated
}

func CreateRandomData(size int) []float64 {

	data := make([]float64, size)
	for i := 0; i < size; i++ {
		rand.Seed(time.Now().UnixNano())
		num := (rand.Float64() * 2) - 1
		data[i] = num
	}

	return data
}

func (t *MatrixStruct[T]) Clone() Matrix[T] {
	return NewMatrix(t.Data, t.Row, t.Col)
}

func (t *MatrixStruct[T]) Transpose() Matrix[T] {
	return NewMatrix(Transpose(t.Data, t.Row, t.Col), t.Col, t.Row)
}

func (t *MatrixStruct[T]) LocalData() []T {
	return t.Data
}

func (t *MatrixStruct[T]) Print() {

	start := 0
	end := t.Col
	for i := 0; i < t.Row; i++ {
		fmt.Println(t.Data[start:end])
		start = end
		end = start + t.Col
	}
}

func (t *MatrixStruct[T]) LocalShape() (int, int) {
	return t.Row, t.Col
}

func (t *MatrixStruct[T]) Size() int {
	return t.Row * t.Col
}

func ValidateShapeFromData[T NumTypes](data []T, row, col int) {

	if len(data) != (row * col) {
		panic("Invalid shape from data")
	}
}
