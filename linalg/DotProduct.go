package linalg

import "sync"

type task[T NumTypes] struct {
	x    int
	vecA []T
	vecB []T
}

type result[T NumTypes] struct {
	x     int
	value T
}

func Worker[T NumTypes](tasks <-chan task[T], results chan<- result[T]) {
	for t := range tasks {
		results <- result[T]{x: t.x, value: DotVec(t.vecA, t.vecB)}
	}
}

func Dot[T NumTypes](matrixA, matrixB Matrix[T]) Matrix[T] {
	rowA, colA := matrixA.LocalShape()
	rowB, colB := matrixB.LocalShape()

	if colA != rowB {
		panic("Invalid matrix shape for dot product")
	}

	mtxRowA := GetRow(matrixA.LocalData(), rowA, colA)
	mtxColB := GetCols(matrixB.LocalData(), rowB, colB)

	data := make([]T, rowA*colB)

	// Create a channel for tasks and a channel for results
	tasks := make(chan task[T], rowA*colB)
	results := make(chan result[T], rowA*colB)

	// Start a fixed number of workers
	numWorkers := 20
	var wg sync.WaitGroup
	wg.Add(numWorkers)
	for w := 0; w < numWorkers; w++ {
		go func() {
			Worker(tasks, results)
			wg.Done()
		}()
	}

	// Add tasks to the task channel
	x := 0
	for i := 0; i < rowA; i++ {
		for y := 0; y < colB; y++ {
			tasks <- task[T]{x: x, vecA: mtxRowA[i], vecB: mtxColB[y]}
			x += 1
		}
	}
	close(tasks)

	// Wait for all workers to finish
	wg.Wait()

	// Collect the results
	for i := 0; i < rowA*colB; i++ {
		result := <-results
		data[result.x] = result.value
	}

	return NewMatrix(data, rowA, colB)
}

func Dot2[T NumTypes](matrixA, matrixB Matrix[T]) Matrix[T] {
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
