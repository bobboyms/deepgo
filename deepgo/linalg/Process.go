package linalg

import (
	"sync"
)

func Subslice[T NumTypes](arr []T, start int, end int) []T {
	//if start < 0 || end > len(arr) || start > end {
	//	panic("invalid start or end index")
	//}

	subslice := make([]T, end-start)
	for i := start; i < end; i++ {
		subslice[i-start] = arr[i]
	}

	return subslice
}

func ProcessOperation[T NumTypes](matrixA, matrixB Matrix[T], operation func(vecA, vecB []T) []T) Matrix[T] {

	rowA, colA := matrixA.LocalShape()
	_, colB := matrixB.LocalShape()

	if colB != colA {
		panic("The matrices must have the same shape.")
	}

	dataA := matrixA.LocalData()
	dataB := matrixB.LocalData()

	//rowDataA := GetRow(matrixA.LocalData(), rowA, colA)
	//rowDataB := GetRow(matrixB.LocalData(), rowB, colB)
	//
	//if len(rowDataA) < 0 {
	//	println("dsd")
	//	println(rowDataB)
	//}

	start := 0
	end := colA
	tempMatrix := make([][]T, rowA)
	var wg sync.WaitGroup
	wg.Add(rowA)
	for i := 0; i < rowA; i++ {

		go func(i, start, end int) {
			defer wg.Done()
			tempMatrix[i] = operation(dataA[start:end], dataB[start:end])
		}(i, start, end)

		start = end
		end = start + colA
	}
	wg.Wait()

	var tempData []T
	for _, temp := range tempMatrix {
		for _, val := range temp {
			tempData = append(tempData, val)
		}
	}

	return NewMatrix(tempData, rowA, colA)
}

func DivVec[T NumTypes](vecA, vecB []T) []T {

	vecResult := make([]T, len(vecA))
	for i := 0; i < len(vecA); i++ {
		vecResult[i] = vecA[i] / vecB[i]
	}
	return vecResult
}

func SubVec[T NumTypes](vecA, vecB []T) []T {

	vecResult := make([]T, len(vecA))
	for i := 0; i < len(vecA); i++ {
		vecResult[i] = vecA[i] - vecB[i]
	}
	return vecResult
}

func SumVec[T NumTypes](vecA, vecB []T) []T {

	vecResult := make([]T, len(vecA))
	for i := 0; i < len(vecA); i++ {
		vecResult[i] = vecA[i] + vecB[i]
	}
	return vecResult
}

func DotVec[T NumTypes](vecA, vecB []T) T {
	var sum T = 0
	for i := 0; i < len(vecA); i++ {
		sum += vecA[i] * vecB[i]
	}
	return sum
}

func MulVec[T NumTypes](vecA, vecB []T) []T {

	vecResult := make([]T, len(vecA))
	for i := 0; i < len(vecA); i++ {
		vecResult[i] = vecA[i] * vecB[i]
	}
	return vecResult
}
