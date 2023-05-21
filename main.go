package main

import (
	"fmt"
	"tensors-processing/deepgo/activation"
	"tensors-processing/deepgo/datasets"
	"tensors-processing/deepgo/linalg"
	"tensors-processing/deepgo/loss"
	"tensors-processing/deepgo/nn"
	"tensors-processing/deepgo/preprocessing"
)

func main() {

	ds := datasets.NewIrisDataSet()
	X, Y := preprocessing.SeparateXY(linalg.NewMatrixFrom2D(ds.GetData(), len(ds.GetData()), len(ds.GetData()[0])))

	XN := preprocessing.NormalizeData(X)
	YN := preprocessing.OneHotEncoder(Y.LocalData())

	xTrain, _, yTrain, _ := datasets.TrainTestSplit(XN, YN, 0.2, 42)

	//w1 := linalg.NewMatrixFrom2D([][]float64{{0.37964287, 0.97761403, 0.70293974}, {0.24318438, 0.89182217, 0.79628823}}, 2, 3)
	layer1 := nn.NewDense(4, 6, activation.Sigmoid)

	//w2 := linalg.NewMatrixFrom2D([][]float64{{0.58468626, 0.24506299}, {0.1914211, 0.9668479}, {0.42943575, 0.3241452}}, 3, 2)
	layer2 := nn.NewDense(6, 4, activation.Sigmoid)

	//w3 := linalg.NewMatrixFrom2D([][]float64{{0.91123983}, {0.12691277}}, 2, 1)
	layer3 := nn.NewDense(4, 4, activation.Sigmoid)

	learningRate := 0.02

	for epoch := 0; epoch < 1000; epoch++ {
		xRow, xCol := xTrain.LocalShape()
		yRow, yCol := yTrain.LocalShape()
		xRows := linalg.GetRow(xTrain.LocalData(), xRow, xCol)
		yRows := linalg.GetRow(yTrain.LocalData(), yRow, yCol)

		totalLoss := 0.0
		for i := range xRows {
			xi := linalg.NewMatrix(xRows[i], 1, xCol)
			yi := linalg.NewMatrix(yRows[i], 1, yCol)

			r1 := layer1.Forward(xi)
			r2 := layer2.Forward(r1)
			output := layer3.Forward(r2)

			//calcula o erro
			lossx := loss.Mse(yi, output)
			totalLoss += lossx

			/////////////////////////////////////////////////////////////////////////////////////
			//Execute a etapa de backpropagation com SGD

			//Calcule os erros na última camada (output layer)
			//O erro é a diferença entre a saída desejada (Y) e a saída atual (output)
			//Aplique a derivada da função de ativação (sigmoid) para obter a taxa de erro
			w3Deltas := linalg.Mul(linalg.Sub(output, yi), activation.SigmoidDerivative(output))

			// Propague o erro para trás, para a camada escondida anterior (segunda camada escondida)
			// O erro é a matriz transposta dos pesos da camada de saída (weights3) multiplicada pelo erro da camada de saída
			// Em seguida, aplique a derivada da função de ativação (sigmoid) para obter a taxa de erro
			// self.weights2_deltas = np.dot(self.weights3_deltas, self.weights3.T) * self.sigmoid_derivative(self.hidden_layer2)
			w2Deltas := linalg.Mul(linalg.Dot(w3Deltas, layer3.W().Transpose()), activation.SigmoidDerivative(r2))

			// Propague o erro para trás, para a camada escondida anterior (primeira camada escondida)
			// O erro é a matriz transposta dos pesos da segunda camada escondida (weights2) multiplicada pelo erro da segunda camada escondida
			// Em seguida, aplique a derivada da função de ativação (sigmoid) para obter a taxa de erro
			// self.weights1_deltas = np.dot(self.weights2_deltas, self.weights2.T) * self.sigmoid_derivative(self.hidden_layer1)
			w1Deltas := linalg.Mul(linalg.Dot(w2Deltas, layer2.W().Transpose()), activation.SigmoidDerivative(r1))

			//# Atualize os pesos da camada de saída (weights3) subtraindo o produto da taxa de erro da camada de saída e a matriz transposta da segunda camada escondida
			// self.weights3 = self.weights3 - learning_rate * np.dot(self.hidden_layer2.T, self.weights3_deltas)
			newWeights3 := linalg.Sub(layer3.W(), linalg.MulScalar(learningRate, linalg.Dot(r2.Transpose(), w3Deltas)))
			layer3.ChangeW(newWeights3)

			//# Atualize os pesos da segunda camada escondida (weights2) subtraindo o produto da taxa de erro da segunda camada escondida e a matriz transposta da primeira camada escondida
			//self.weights2 -= np.dot(self.hidden_layer1.T, self.weights2_deltas)
			//newWeights2 := linalg.Sub(layer2.W(), linalg.Dot(r1.Transpose(), w2Deltas))
			newWeights2 := linalg.Sub(layer2.W(), linalg.MulScalar(learningRate, linalg.Dot(r1.Transpose(), w2Deltas)))
			layer2.ChangeW(newWeights2)

			// Atualize os pesos da primeira camada escondida (weights1) subtraindo o produto da taxa de erro da primeira camada escondida e a matriz transposta da entrada (X)
			// self.weights1 -= np.dot(X.T, self.weights1_deltas)
			//newWeights1 := linalg.Sub(layer1.W(), linalg.Dot(xi.Transpose(), w1Deltas))
			newWeights1 := linalg.Sub(layer1.W(), linalg.MulScalar(learningRate, linalg.Dot(xi.Transpose(), w1Deltas)))
			layer1.ChangeW(newWeights1)

		}

		meanLoss := totalLoss / float64(xRow)
		if epoch%10 == 0 {
			fmt.Printf("Epoch: %d, Loss: %f\n", epoch, meanLoss)
		}

	}

}
