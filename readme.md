
# Introduction

This code represents the construction of a multi-layer neural network written in Go (golang). It's a complete implementation. The training is carried out via backpropagation through Stochastic Gradient Descent (SGD).

## Dataset

In this example, we will use the Iris dataset, widely known in the machine learning field. The code below is organized in such a way that the data are initially separated into two distinct sets: one for training and another for testing.

The code starts with the creation of the Iris dataset using the command ds := datasets.NewIrisDataSet(). After this, the inputs (flower features) and outputs (flower species) are separated using the preprocessing.SeparateXY function. Subsequently, this data is normalized and preprocessed to be used in the neural network. For the inputs, we use preprocessing.NormalizeData(X) to scale the data to a standard range. For the outputs, the preprocessing.OneHotEncoder(Y.LocalData()) function is used to transform the class labels into a format suitable for training neural networks.

Finally, the datasets.TrainTestSplit(XN, YN, 0.2, 42) function splits the dataset into a training set and a test set, reserving 20% of the data for the test set.

```Go
ds := datasets.NewIrisDataSet()
X, Y := preprocessing.SeparateXY(linalg.NewMatrixFrom2D(ds.GetData(), len(ds.GetData()), len(ds.GetData()[0])))

XN := preprocessing.NormalizeData(X)
YN := preprocessing.OneHotEncoder(Y.LocalData())

xTrain, xTest, yTrain, yTest := datasets.TrainTestSplit(XN, YN, 0.2, 42)
```

## Build Neural Network

```Go


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