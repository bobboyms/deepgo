package main

import (
	"fmt"
	"math/rand"
	"tensors-processing/deepgo/activation"
	"tensors-processing/deepgo/datasets"
	"tensors-processing/deepgo/linalg"
	"tensors-processing/deepgo/loss"
	"tensors-processing/deepgo/metrics"
	"tensors-processing/deepgo/nn"
	"tensors-processing/deepgo/preprocessing"
	"tensors-processing/deepgo/regularization"
	"time"
)

func main() {

	start := time.Now()

	rand.Seed(0)

	ds := datasets.NewIrisDataSet()
	X, Y := preprocessing.SeparateXY(linalg.NewMatrixFrom2D(ds.GetData(), len(ds.GetData()), len(ds.GetData()[0])))

	XN := preprocessing.NormalizeData(X)
	YN := preprocessing.OneHotEncoder(Y.LocalData())

	xTrain, xTest, yTrain, yTest := datasets.TrainTestSplit(XN, YN, 0.2, 42)

	//fmt.Println(xTrain.Size())

	layer1 := nn.NewDense(4, 7, activation.Sigmoid)
	layer2 := nn.NewDense(7, 8, activation.Sigmoid)
	layer3 := nn.NewDense(8, 4, activation.Sigmoid)

	learningRate := 0.01
	//l2Lambda := 0.001

	xRow, xCol := xTrain.LocalShape()
	yRow, yCol := yTrain.LocalShape()

	xRows := preprocessing.CreateBatches(linalg.GetRow(xTrain.LocalData(), xRow, xCol), 50)
	yRows := preprocessing.CreateBatches(linalg.GetRow(yTrain.LocalData(), yRow, yCol), 50)

	for epoch := 0; epoch < 110; epoch++ {

		totalLoss := 0.0
		for i := range xRows {
			xi := linalg.NewMatrixFrom2D(xRows[i], len(xRows[i]), xCol)
			yi := linalg.NewMatrixFrom2D(yRows[i], len(yRows[i]), yCol)

			r1 := layer1.Forward(xi)
			r2 := layer2.Forward(regularization.Dropout(r1, 0.15))
			output := layer3.Forward(r2)

			//calcula o erro
			totalLoss += loss.Mse(yi, output)
			//totalLoss += lossx

			/////////////////////////////////////////////////////////////////////////////////////
			//Execute a etapa de backpropagation com SGD

			//Calcule os erros na última camada (output layer)
			//O erro é a diferença entre a saída desejada (Y) e a saída atual (output)
			//Aplique a derivada da função de ativação (sigmoid) para obter a taxa de erro
			w3Deltas := linalg.Mul(linalg.Sub(output, yi), activation.SigmoidDerivative(output))

			// Propague o erro para trás, para a camada escondida anterior (segunda camada escondida)
			// O erro é a matriz transposta dos pesos da camada de saída (weights3) multiplicada pelo erro da camada de saída
			// Em seguida, aplique a derivada da função de ativação (sigmoid) para obter a taxa de erro
			w2Deltas := linalg.Mul(linalg.Dot(w3Deltas, layer3.W().Transpose()), activation.SigmoidDerivative(r2))

			// Propague o erro para trás, para a camada escondida anterior (primeira camada escondida)
			// O erro é a matriz transposta dos pesos da segunda camada escondida (weights2) multiplicada pelo erro da segunda camada escondida
			// Em seguida, aplique a derivada da função de ativação (sigmoid) para obter a taxa de erro
			w1Deltas := linalg.Mul(linalg.Dot(w2Deltas, layer2.W().Transpose()), activation.SigmoidDerivative(r1))

			//Atualiza os pesos
			newWeights3 := linalg.Sub(layer3.W(), linalg.MulScalar(learningRate, linalg.Dot(r2.Transpose(), w3Deltas)))
			layer3.ChangeW(newWeights3)

			newWeights2 := linalg.Sub(layer2.W(), linalg.MulScalar(learningRate, linalg.Dot(r1.Transpose(), w2Deltas)))
			layer2.ChangeW(newWeights2)

			newWeights1 := linalg.Sub(layer1.W(), linalg.MulScalar(learningRate, linalg.Dot(xi.Transpose(), w1Deltas)))
			layer1.ChangeW(newWeights1)

			//Atualiza os biases
			newBiases3 := linalg.Sub(layer3.B(), linalg.MulScalar(learningRate, linalg.SumAxis(w3Deltas, 0)))
			layer3.ChangeB(newBiases3)
			newBiases2 := linalg.Sub(layer2.B(), linalg.MulScalar(learningRate, linalg.SumAxis(w2Deltas, 0)))
			layer2.ChangeB(newBiases2)
			newBiases1 := linalg.Sub(layer1.B(), linalg.MulScalar(learningRate, linalg.SumAxis(w1Deltas, 0)))
			layer1.ChangeB(newBiases1)

		}

		meanLoss := totalLoss / float64(len(xRows))
		if epoch%10 == 0 {
			fmt.Printf("Epoch: %d, Loss: %f\n", epoch, meanLoss)
		}

	}

	r1 := layer1.Forward(xTest)
	r2 := layer2.Forward(r1)
	outputs := layer3.Forward(r2)

	fmt.Printf("Accuracy: %f\n", metrics.Accuracy(yTest, outputs))

	elapsed := time.Since(start)
	fmt.Println("Tempo execução: ", elapsed)

}
