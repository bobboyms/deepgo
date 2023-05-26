package tensors

import (
	"fmt"
	"reflect"
)

func Transpose2D(arr interface{}) ([][]interface{}, error) {
	v := reflect.ValueOf(arr)

	// Check if the input is a slice.
	if v.Kind() != reflect.Slice {
		return nil, fmt.Errorf("input is not a slice")
	}

	// Get the length of the inner slices.
	if v.Len() == 0 {
		return [][]interface{}{}, nil
	}

	if v.Index(0).Kind() != reflect.Slice {
		return nil, fmt.Errorf("inner element is not a slice")
	}

	innerLen := v.Index(0).Len()

	// Transpose
	result := make([][]interface{}, innerLen)
	for i := 0; i < innerLen; i++ {
		result[i] = make([]interface{}, v.Len())
		for j := 0; j < v.Len(); j++ {
			if v.Index(j).Len() != innerLen {
				return nil, fmt.Errorf("inner slices are not of the same length")
			}
			result[i][j] = v.Index(j).Index(i).Interface()
		}
	}

	return result, nil
}

func Transpose2DFloat64(matrix [][]float64, shape []int) [][]float64 {
	rows := shape[0]
	cols := shape[1]

	transposed := make([][]float64, cols)
	for i := range transposed {
		transposed[i] = make([]float64, rows)
	}

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			transposed[j][i] = matrix[i][j]
		}
	}

	return transposed
}
