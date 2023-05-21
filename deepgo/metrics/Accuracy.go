package metrics

import (
	"tensors-processing/deepgo/linalg"
	"tensors-processing/deepgo/preprocessing"
)

func Accuracy(yTrue, yPred linalg.Matrix[float64]) float64 {

	ytRow, ytCol := yTrue.LocalShape()
	ypRow, ypCol := yPred.LocalShape()

	if ytRow != ypRow {
		panic("Os vetores yTrue e yPred devem ter o mesmo comprimento.")
	}

	ytData := linalg.GetRow(yTrue.LocalData(), ytRow, ytCol)
	ypData := linalg.GetRow(yPred.LocalData(), ypRow, ypCol)

	correctCount := 0
	for i := range ytData {
		if len(ytData[i]) != len(ypData[i]) {
			panic("Os vetores yTrue[i] e yPred[i] devem ter o mesmo comprimento.")
		}

		if preprocessing.OneHotDecode(ytData[i]) == preprocessing.OneHotDecode(ypData[i]) {
			correctCount++
		}
	}

	accuracy := float64(correctCount) / float64(len(ytData))
	return accuracy
}
