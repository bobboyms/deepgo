package tensors

import (
	"fmt"
	"reflect"
)

type TensorType string

const (
	TensorFloat64 TensorType = "float64"
)

type Tensor interface {
	Shape() []int
	Get2DataFloat64() [][]float64
	GetDim() int
	GetTensorType() TensorType
	Transpose() Tensor
}

type TensorStruct struct {
	TensorType TensorType
	LocalData  interface{}
	LocalDim   int
	LocalShape []int
}

func (t TensorStruct) Transpose() Tensor {
	data, _ := Transpose2D(t.LocalData)
	return NewTensor(data, TensorFloat64)
}

func (t TensorStruct) GetTensorType() TensorType {
	return t.TensorType
}

func (t TensorStruct) GetDim() int {
	return t.LocalDim
}

func (t TensorStruct) Get2DataFloat64() [][]float64 {

	if t.TensorType != TensorFloat64 {
		panic(fmt.Sprintf("Invalid get data %s to %s", t.TensorType, TensorFloat64))
	}

	if t.LocalDim != 2 {
		panic(fmt.Sprintf("Invalid conversion to 2D. Shape of tensor: %d", t.LocalDim))
	}

	data, _ := to2DFloat64(t.LocalData)
	return data
}

func NewTensor(data interface{}, tensorType TensorType) Tensor {
	shape := GetShape(data)
	return TensorStruct{
		LocalDim:   len(shape),
		LocalData:  data,
		LocalShape: shape,
		TensorType: tensorType,
	}
}

func (t TensorStruct) Shape() []int {
	return GetShape(t.LocalData)
}

func GetShape(arr interface{}) []int {
	v := reflect.ValueOf(arr)

	if v.Kind() != reflect.Slice {
		return nil
	}

	shape := []int{v.Len()}
	if v.Len() > 0 && v.Index(0).Kind() == reflect.Slice {
		shape = append(shape, GetShape(v.Index(0).Interface())...)
	}

	return shape
}

func to2DFloat64(arr interface{}) ([][]float64, bool) {
	v := reflect.ValueOf(arr)
	if v.Kind() != reflect.Slice {
		return nil, false
	}

	result := make([][]float64, v.Len())

	for i := 0; i < v.Len(); i++ {
		slice := v.Index(i)
		if slice.Kind() != reflect.Slice {
			return nil, false
		}

		innerSlice := make([]float64, slice.Len())
		for j := 0; j < slice.Len(); j++ {
			if value, ok := slice.Index(j).Interface().(float64); ok {
				innerSlice[j] = value
			} else {
				return nil, false
			}
		}
		result[i] = innerSlice
	}

	return result, true
}
