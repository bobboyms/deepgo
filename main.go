package main

import (
	"tensors-processing/activation"
	"tensors-processing/linalg"
	"tensors-processing/nn"
)

func CreateArr(a int) []float64 {
	var x []float64
	for i := 0; i < a; i++ {
		x = append(x, float64(i))
	}
	return x
}

func main() {

	input := linalg.NewMatrix([]float64{2, 4, 6}, 1, 3)

	rnn := nn.NewSequential([]nn.Layer{
		nn.NewDense(3, 4, activation.None),
		nn.NewDense(4, 8, activation.None),
		nn.NewDense(8, 2, activation.None),
	})

	rnn.Predict(input).Print()

	//newTensorA := linalg.NewMatrix([]float32{
	//	2, 2}, 2, 1)
	//newTensorB := linalg.NewMatrix([]int{1, 2, 3, 4}, 2, 2)
	//
	//newTensorC := linalg.NewMatrix([]float32{3, 3}, 1, 2)
	//
	//r := linalg.Dot(newTensorA.Transpose(), cast.CastToFloat32(newTensorB))
	//linalg.Sum(r, newTensorC).Print()

	//var sum int64 = 0
	//for i := 0; i < 100; i++ {
	//	start := time.Now()
	//	linalg.Dot(newTensorA.Transpose(), newTensorB)
	//	elapsed := time.Since(start)
	//	sum += elapsed.Milliseconds()
	//}
	//fmt.Println(sum / 100)
	//
	//
	//sum = 0
	//for i := 0; i < 100; i++ {
	//	start := time.Now()
	//	linalg.DotX(newTensorA, newTensorB)
	//	elapsed := time.Since(start)
	//	sum += elapsed.Milliseconds()
	//}
	//fmt.Println(sum / 100)

}
