package main

import (
	"log"
	"math"
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

func crossEntropy(yTrue, yPred []float64) float64 {
	if len(yTrue) != len(yPred) {
		log.Fatalf("yTrue e yPred devem ter o mesmo tamanho, mas têm tamanhos %d e %d", len(yTrue), len(yPred))
	}

	var loss float64
	for i := range yTrue {
		loss += yTrue[i] * math.Log(yPred[i])
	}

	return -loss
}

func main() {

	input := linalg.NewMatrix([]float64{5.2, 4, 6}, 1, 3)

	rnn := nn.NewSequential([]nn.Layer{
		nn.NewDense(3, 4),
		nn.NewDense(4, 8),
		nn.NewDense(8, 1),
		activation.NewReLU(),
	})

	rnn.Predict(input).Print()
	//
	//linalg.Log(input).Print()
	//
	//newTensorA := linalg.NewMatrix([]float64{0.7, 0.1, 0.2}, 1, 3)
	//newTensorB := linalg.NewMatrix([]float64{1, 0, 0}, 1, 3)
	//
	//fmt.Println(loss.CrossEntropy(newTensorB, newTensorA))

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
