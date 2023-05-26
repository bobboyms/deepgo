package test

import (
	"math/rand"
	"reflect"
	"tensors-processing/deepgo/activation"
	"tensors-processing/deepgo/datasets"
	"tensors-processing/deepgo/linalg"
	"tensors-processing/deepgo/nn"
	"tensors-processing/deepgo/preprocessing"
	"testing"
)

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

func TestGetShape(t *testing.T) {
	rand.Seed(0)

	ds := datasets.NewIrisDataSet()
	X, Y := preprocessing.SeparateXY(linalg.NewMatrixFrom2D(ds.GetData(), len(ds.GetData()), len(ds.GetData()[0])))

	XN := preprocessing.NormalizeData(X)
	YN := preprocessing.OneHotEncoder(Y.LocalData())

	//xTrain, xTest, yTrain, yTest
	xTrain, _, _, _ := datasets.TrainTestSplit(XN, YN, 0.2, 42)

	//row, _ := xTrain.LocalShape()
	//
	//w := linalg.NormalDistribution(row, 4)
	//dotResult := linalg.Dot(xTrain, w.Transpose())
	//
	//_, row = dotResult.LocalShape()
	//b := linalg.NormalDistribution(1, row)
	//activation.Sigmoid(linalg.Sum(dotResult, linalg.NewBroadcasting(b, row))).Print()

	xx := nn.NewDense(4, 6, activation.Sigmoid)
	xx.Forward(xTrain)
}
