package test

import (
	"tensors-processing/deepgo/linalg"
	"tensors-processing/deepgo/preprocessing"
	"testing"
)

func TestOneHotEncoder(t *testing.T) {
	r := preprocessing.OneHotEncoder([]int{1, 2, 3, 2, 1})
	r.Print()
}

func TestStandardScaler(t *testing.T) {
	//[ 1.06904497, -0.39223227],
	//	[-1.33630621, -0.98058068],
	//	[ 0.26726124,  1.37281295]])
	r := preprocessing.NormalizeData(linalg.NewMatrixFrom2D([][]float64{{11, 2}, {5, 1}, {9, 5}}, 3, 2))
	r.Print()

	//fmt.Println(preprocessing.Standardize([][]float64{{11, 2.5}, {5, 1}, {9, 5.2}}))
}
