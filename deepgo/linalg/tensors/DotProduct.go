package tensors

//func Dot2DFloat64(matrixA, matrixB Tensor) Tensor {
//	rowA := matrixA.Shape()[0]
//	colA := matrixA.Shape()[1]
//
//	rowB := matrixB.Shape()[0]
//	colB := matrixB.Shape()[1]
//
//	if colA != rowB {
//		panic("Invalid matrix shape for dot product")
//	}
//
//	mtxRowA := matrixA.Get2DataFloat64()
//	mtxColB := matrixA.Transpose().Get2DataFloat64()
//
//	//x := 0
//	//data := make([][]float64, rowA)
//	//var wg sync.WaitGroup
//	//wg.Add(colB * rowA)
//	for i := 0; i < rowA; i++ {
//		//temp := make([]float64, colB)
//		//for y := 0; y < colB; y++ {
//		//	go func(x, i, y int, vecA, vecB []float64) {
//		//		temp[y] = DotVecFloat64(vecA, vecB)
//		//	}(x, i, y, mtxRowA[i], mtxColB[y])
//		//	x += 1
//		//}
//		//data[i] = temp
//	}
//	wg.Wait()
//
//	return nil
//}

func DotVecFloat64(vecA, vecB []float64) float64 {
	sum := 0.0
	for i := 0; i < len(vecA); i++ {
		sum += vecA[i] * vecB[i]
	}
	return sum
}
