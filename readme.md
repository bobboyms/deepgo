This code represents the construction of a multi-layer neural network written in Go (golang). It's a complete implementation. The training is carried out via backpropagation through Stochastic Gradient Descent (SGD).

```Go
ds := datasets.NewIrisDataSet()
	X, Y := preprocessing.SeparateXY(linalg.NewMatrixFrom2D(ds.GetData(), len(ds.GetData()), len(ds.GetData()[0])))

	XN := preprocessing.NormalizeData(X)
	YN := preprocessing.OneHotEncoder(Y.LocalData())

	xTrain, xTest, yTrain, yTest := datasets.TrainTestSplit(XN, YN, 0.2, 42)

	layer1 := nn.NewDense(4, 6, activation.Sigmoid)
	layer2 := nn.NewDense(6, 4, activation.Sigmoid)
	layer3 := nn.NewDense(4, 4, activation.Sigmoid)

	learningRate := 0.6

	for epoch := 0; epoch < 150; epoch++ {
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
			w2Deltas := linalg.Mul(linalg.Dot(w3Deltas, layer3.W().Transpose()), activation.SigmoidDerivative(r2))

			// Propague o erro para trás, para a camada escondida anterior (primeira camada escondida)
			// O erro é a matriz transposta dos pesos da segunda camada escondida (weights2) multiplicada pelo erro da segunda camada escondida
			// Em seguida, aplique a derivada da função de ativação (sigmoid) para obter a taxa de erro
			w1Deltas := linalg.Mul(linalg.Dot(w2Deltas, layer2.W().Transpose()), activation.SigmoidDerivative(r1))

			//# Atualize os pesos da camada de saída (weights3) subtraindo o produto da taxa de erro da camada de saída e a matriz transposta da segunda camada escondida
			// self.weights3 = self.weights3 - learning_rate * np.dot(self.hidden_layer2.T, self.weights3_deltas)
			newWeights3 := linalg.Sub(layer3.W(), linalg.MulScalar(learningRate, linalg.Dot(r2.Transpose(), w3Deltas)))
			layer3.ChangeW(newWeights3)

			//# Atualize os pesos da segunda camada escondida (weights2) subtraindo o produto da taxa de erro da segunda camada escondida e a matriz transposta da primeira camada escondida
			newWeights2 := linalg.Sub(layer2.W(), linalg.MulScalar(learningRate, linalg.Dot(r1.Transpose(), w2Deltas)))
			layer2.ChangeW(newWeights2)

			// Atualize os pesos da primeira camada escondida (weights1) subtraindo o produto da taxa de erro da primeira camada escondida e a matriz transposta da entrada (X)
			newWeights1 := linalg.Sub(layer1.W(), linalg.MulScalar(learningRate, linalg.Dot(xi.Transpose(), w1Deltas)))
			layer1.ChangeW(newWeights1)

		}

		meanLoss := totalLoss / float64(xRow)
		if epoch%10 == 0 {
			fmt.Printf("Epoch: %d, Loss: %f\n", epoch, meanLoss)
		}

	}

	r1 := layer1.Forward(xTest)
	r2 := layer2.Forward(r1)
	output := layer3.Forward(r2)

	fmt.Println("-------------------")
	fmt.Printf("Accuracy: %f", metrics.Accuracy(yTest, output))
```