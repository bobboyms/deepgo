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

func (i *IrisDataSet) TrainTestSplit(testSize float64, randomState int64) (linalg.Matrix[float64], linalg.Matrix[float64], linalg.Matrix[float64], linalg.Matrix[float64]) {
	xTrain, xTest, yTrain, yTest := TrainTestSplit(i.GetData(), testSize, randomState)
	return linalg.NewMatrixFrom2D(xTrain, len(xTrain), 4),
		linalg.NewMatrixFrom2D(xTest, len(xTest), 4),
		linalg.NewMatrix(yTrain, len(yTrain), 1),
		linalg.NewMatrix(yTest, len(yTest), 1)
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

func TrainTestSplit(values [][]float64, testSize float64, randomState int64) ([][]float64, [][]float64, []float64, []float64) {

	if testSize >= 1.0 {
		panic("O tamanho do conjunto de teste deve ser menor que 1.0")
	}

	r := rand.New(rand.NewSource(randomState))
	shuffled := make([][]float64, len(values))
	copy(shuffled, values) // Copia os valores originais para shuffled

	for i := range shuffled {
		j := r.Intn(i + 1)
		shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
	}

	nTest := int(testSize * float64(len(shuffled)))
	if nTest >= len(shuffled) {
		panic("O tamanho do conjunto de teste é maior ou igual ao tamanho dos dados originais")
	}

	xTrain := make([][]float64, len(shuffled)-nTest)
	yTrain := make([]float64, len(shuffled)-nTest)
	for i, cell := range shuffled[:len(shuffled)-nTest] {
		xTrain[i] = make([]float64, 4)
		copy(xTrain[i], cell[:4])
		yTrain[i] = cell[4]
	}

	xTest := make([][]float64, nTest)
	yTest := make([]float64, nTest)
	for i, cell := range shuffled[len(shuffled)-nTest:] {
		xTest[i] = make([]float64, 4)
		copy(xTest[i], cell[:4])
		yTest[i] = cell[4]
	}

	return xTrain, xTest, yTrain, yTest
}
