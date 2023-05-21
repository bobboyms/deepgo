package test

import (
	"tensors-processing/deepgo/datasets"
	"tensors-processing/deepgo/linalg"
	"tensors-processing/deepgo/preprocessing"
	"testing"
)

func TestOneHotEncoder(t *testing.T) {
	r := preprocessing.OneHotEncoder([]int{1, 2, 3, 2, 1})
	r.Print()
}

func TestSeparateXY(t *testing.T) {

	ds := datasets.NewIrisDataSet()
	X, Y := preprocessing.SeparateXY(linalg.NewMatrixFrom2D(ds.GetData(), len(ds.GetData()), len(ds.GetData()[0])))

	XN := preprocessing.NormalizeData(X)
	YN := preprocessing.OneHotEncoder(Y.LocalData())

	yTrain, _, _, _ := datasets.TrainTestSplit(XN, YN, 0.2, 42)

	yTrain.Print()

}

func TestStandardScaler(t *testing.T) {

	r := preprocessing.NormalizeData(linalg.NewMatrixFrom2D([][]float64{{11, 2}, {5, 1}, {9, 5}}, 3, 2))
	r.Print()

}
