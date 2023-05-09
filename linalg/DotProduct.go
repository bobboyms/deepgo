package linalg

import "sync"

func Dot[T NumTypes](matrixA, matrixB Matrix[T]) Matrix[T] {
	rowA, colA := matrixA.LocalShape()
	rowB, colB := matrixB.LocalShape()

	if colA != rowB {
		panic("Invalid matrix shape for dot product")
	}

	mtxRowA := GetRow(matrixA.LocalData(), rowA, colA)
	mtxColB := GetCols(matrixB.LocalData(), rowB, colB)

	x := 0
	data := make([]T, rowA*colB)
	var wg sync.WaitGroup
	wg.Add(colB * rowA)
	for i := 0; i < rowA; i++ {
		for y := 0; y < colB; y++ {
			go func(x, i, y int, vecA, vecB []T) {
				defer wg.Done()
				data[x] = DotVec(vecA, vecB)
			}(x, i, y, mtxRowA[i], mtxColB[y])
			x += 1
		}
	}
	wg.Wait()
	return NewMatrix(data, rowA, colB)
}

func GetCols[T NumTypes](matrix []T, row, col int) [][]T {
	tempRowsMtx := GetRow(matrix, row, col)
	var tempMatrix [][]T
	for i := 0; i < col; i++ {
		var newData []T
		for y := 0; y < row; y++ {
			newData = append(newData, tempRowsMtx[y][i])
		}
		tempMatrix = append(tempMatrix, newData)
	}

	return tempMatrix
}

func GetRow[T NumTypes](matrix []T, row, col int) [][]T {
	start := 0
	end := col
	var tempMatrix [][]T
	for i := 0; i < row; i++ {
		tempMatrix = append(tempMatrix, matrix[start:end])
		start = end
		end = start + col
	}
	return tempMatrix
}
