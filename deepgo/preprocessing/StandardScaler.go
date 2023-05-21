package preprocessing

import (
	"math"
	"tensors-processing/deepgo/linalg"
)

// Function to calculate mean
func mean(data []float64) float64 {
	sum := 0.0
	for _, value := range data {
		sum += value
	}
	return sum / float64(len(data))
}

// Function to calculate standard deviation
func stdDev(data []float64, mean float64) float64 {
	sum := 0.0
	for _, value := range data {
		diff := value - mean
		sum += diff * diff
	}
	variance := sum / float64(len(data)-1)
	return math.Sqrt(variance)
}

// Function to normalize data
func NormalizeData(data linalg.Matrix[float64]) linalg.Matrix[float64] {

	nRow, nCol := data.LocalShape()
	rows := linalg.GetRow(data.LocalData(), nRow, nCol)

	meanList := make([]float64, len(rows[0]))
	stdDevList := make([]float64, len(rows[0]))

	for i := range rows[0] {
		tempColumn := make([]float64, len(rows))
		for j := range rows {
			tempColumn[j] = rows[j][i]
		}
		meanList[i] = mean(tempColumn)
		stdDevList[i] = stdDev(tempColumn, meanList[i])
	}

	normalizedData := make([][]float64, len(rows))
	for i := range rows {
		normalizedData[i] = make([]float64, len(rows[i]))
		for j := range rows[i] {
			normalizedData[i][j] = (rows[i][j] - meanList[j]) / stdDevList[j]
		}
	}

	return linalg.NewMatrixFrom2D(normalizedData, nRow, nCol)
}

func findMinMax(arr [][]float64) (float64, float64) {
	min := arr[0][0]
	max := arr[0][0]

	for _, subArr := range arr {
		for _, val := range subArr {
			if val < min {
				min = val
			}
			if val > max {
				max = val
			}
		}
	}

	return min, max
}

func SeparateXY(matrix linalg.Matrix[float64]) (linalg.Matrix[float64], linalg.Matrix[float64]) {
	numRows, numCols := matrix.LocalShape()
	data := linalg.GetRow(matrix.LocalData(), numRows, numCols)

	X := make([][]float64, numRows)
	Y := make([]float64, numRows)

	for i := 0; i < numRows; i++ {
		X[i] = make([]float64, numCols-1)
		for j := 0; j < numCols-1; j++ {
			X[i][j] = data[i][j]
		}
		Y[i] = data[i][numCols-1]
	}

	return linalg.NewMatrixFrom2D(X, numRows, numCols-1), linalg.NewMatrix(Y, numRows, 1)
}
