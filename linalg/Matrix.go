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

type TensorStruct[T NumTypes] struct {
	Data []T
	Row  int
	Col  int
}

type NumTypes interface {
	int | int8 | int16 | int32 | int64 | float64 | float32
}

func NewMatrix[T NumTypes](data []T, row, col int) Matrix[T] {
	ValidateShapeFromData(data, row, col)
	return &TensorStruct[T]{
		Data: data,
		Row:  row,
		Col:  col,
	}
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

func (t *TensorStruct[T]) Clone() Matrix[T] {
	return NewMatrix(t.Data, t.Row, t.Col)
}

func (t *TensorStruct[T]) Transpose() Matrix[T] {
	return NewMatrix(Transpose(t.Data, t.Row, t.Col), t.Col, t.Row)
}

func (t *TensorStruct[T]) LocalData() []T {
	return t.Data
}

func (t *TensorStruct[T]) Print() {

	start := 0
	end := t.Col
	for i := 0; i < t.Row; i++ {
		fmt.Println(t.Data[start:end])
		start = end
		end = start + t.Col
	}
}

func (t *TensorStruct[T]) LocalShape() (int, int) {
	return t.Row, t.Col
}

func (t *TensorStruct[T]) Size() int {
	return t.Row * t.Col
}

func ValidateShapeFromData[T NumTypes](data []T, row, col int) {

	if len(data) != (row * col) {
		panic("Invalid shape from data")
	}
}
