package datasets

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math/rand"
	"net/http"
	"tensors-processing/deepgo/linalg"
)

type Iris struct {
	SepalLength float64 `json:"sepalLength"`
	SepalWidth  float64 `json:"sepalWidth"`
	PetalLength float64 `json:"petalLength"`
	PetalWidth  float64 `json:"petalWidth"`
	Species     string  `json:"species"`
}

type IrisDataSet struct {
	RawData []Iris
}

func NewIrisDataSet() IrisDataSet {
	return IrisDataSet{
		RawData: NewIris(),
	}
}

func (i *IrisDataSet) GetData() [][]float64 {
	var values [][]float64
	for _, iris := range i.RawData {
		var temp []float64
		temp = append(temp, iris.SepalLength)
		temp = append(temp, iris.SepalWidth)
		temp = append(temp, iris.PetalLength)
		temp = append(temp, iris.PetalWidth)
		temp = append(temp, GetIdClass(iris.Species))
		values = append(values, temp)
	}

	return values
}


func GetIdClass(value string) float64 {

	if value == "setosa" {
		return 1
	}

	if value == "versicolor" {
		return 2
	}

	if value == "virginica" {
		return 3
	}

	panic("Class not found")
}

func NewIris() []Iris {

	url := "https://raw.githubusercontent.com/domoritz/maps/master/data/iris.json"

	response, err := http.Get(url)
	if err != nil {
		fmt.Println("Erro ao fazer a solicitação HTTP:", err)
		return nil
	}
	defer response.Body.Close()

	body, err := ioutil.ReadAll(response.Body)
	if err != nil {
		fmt.Println("Erro ao ler o corpo da resposta:", err)
		return nil
	}

	var irisData []Iris
	err = json.Unmarshal(body, &irisData)
	if err != nil {
		fmt.Println("Erro ao decodificar o JSON:", err)
		return nil
	}

	return irisData
}

//func TrainTestSplit(values [][]float64, testSize float64, randomState int64) ([][]float64, [][]float64, []float64, []float64) {
//
//	if testSize >= 1.0 {
//		panic("O tamanho do conjunto de teste deve ser menor que 1.0")
//	}
//
//	r := rand.New(rand.NewSource(randomState))
//	shuffled := make([][]float64, len(values))
//	copy(shuffled, values) // Copia os valores originais para shuffled
//
//	for i := range shuffled {
//		j := r.Intn(i + 1)
//		shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
//	}
//
//	nTest := int(testSize * float64(len(shuffled)))
//	if nTest >= len(shuffled) {
//		panic("O tamanho do conjunto de teste é maior ou igual ao tamanho dos dados originais")
//	}
//
//	xTrain := make([][]float64, len(shuffled)-nTest)
//	yTrain := make([]float64, len(shuffled)-nTest)
//	for i, cell := range shuffled[:len(shuffled)-nTest] {
//		xTrain[i] = make([]float64, 4)
//		copy(xTrain[i], cell[:4])
//		yTrain[i] = cell[4]
//	}
//
//	xTest := make([][]float64, nTest)
//	yTest := make([]float64, nTest)
//	for i, cell := range shuffled[len(shuffled)-nTest:] {
//		xTest[i] = make([]float64, 4)
//		copy(xTest[i], cell[:4])
//		yTest[i] = cell[4]
//	}
//
//	return xTrain, xTest, yTrain, yTest
//}

//func TrainTestSplit(X, Y [][]float64, testSize float64, randomState int64) ([][]float64, [][]float64, [][]float64, [][]float64) {
//	if len(X) != len(Y) {
//		panic("O número de linhas em X deve ser igual ao número de linhas em Y.")
//	}
//
//	if testSize >= 1.0 {
//		panic("O tamanho do conjunto de teste deve ser menor que 1.0")
//	}
//
//	r := rand.New(rand.NewSource(randomState))
//
//	for i := range X {
//		j := r.Intn(i + 1)
//		X[i], X[j] = X[j], X[i]
//		Y[i], Y[j] = Y[j], Y[i]
//	}
//
//	nTest := int(testSize * float64(len(X)))
//	if nTest >= len(X) {
//		panic("O tamanho do conjunto de teste é maior ou igual ao tamanho dos dados originais")
//	}
//
//	xTrain := X[:len(X)-nTest]
//	yTrain := Y[:len(Y)-nTest]
//
//	xTest := X[len(X)-nTest:]
//	yTest := Y[len(Y)-nTest:]
//
//	return xTrain, xTest, yTrain, yTest
//}

func TrainTestSplit(X, Y linalg.Matrix[float64], testSize float64, randomState int64) (linalg.Matrix[float64], linalg.Matrix[float64], linalg.Matrix[float64], linalg.Matrix[float64]) {

	xRow, xCol := X.LocalShape()
	yRow, yCol := Y.LocalShape()

	if xRow != yRow {
		panic("O número de linhas em X deve ser igual ao número de linhas em Y.")
	}

	if testSize >= 1.0 {
		panic("O tamanho do conjunto de teste deve ser menor que 1.0")
	}

	xData := linalg.GetRow(X.LocalData(), xRow, xCol)
	yData := linalg.GetRow(Y.LocalData(), yRow, yCol)

	r := rand.New(rand.NewSource(randomState))

	for i := range xData {
		j := r.Intn(i + 1)
		xData[i], xData[j] = xData[j], xData[i]
		yData[i], yData[j] = yData[j], yData[i]
	}

	nTest := int(testSize * float64(xRow))
	if nTest >= xRow {
		panic("O tamanho do conjunto de teste é maior ou igual ao tamanho dos dados originais")
	}

	xTrain := xData[:xRow-nTest]
	yTrain := yData[:yRow-nTest]

	xTest := xData[xRow-nTest:]
	yTest := yData[yRow-nTest:]

	return linalg.NewMatrixFrom2D(xTrain, len(xTrain), xCol),
		linalg.NewMatrixFrom2D(xTest, len(xTest), xCol),
		linalg.NewMatrixFrom2D(yTrain, len(yTrain), yCol),
		linalg.NewMatrixFrom2D(yTest, len(yTest), yCol)
}
