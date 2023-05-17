package main

import (
	"log"
	"math"
	"tensors-processing/linalg"
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

	//input := linalg.NewMatrix([]float64{5.2, 4, 6}, 1, 3)
	//
	//rnn := nn.NewSequential([]nn.Layer{
	//	nn.NewDense(3, 4),
	//	activation.NewSigmoid(),
	//	nn.NewDense(4, 80),
	//	activation.NewSigmoid(),
	//	nn.NewDense(80, 5),
	//	activation.NewSoftmax(),
	//})
	//rnn.Predict(input).Print()
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

	newTensorA := linalg.NewMatrix(CreateArr(5*3), 5, 3)
	newTensorB := linalg.NewMatrix(CreateArr(5*3), 5, 3)

	//start := time.Now()
	linalg.Dot(newTensorA.Transpose(), newTensorB).Print()
	println("++++")
	linalg.Dot2(newTensorA.Transpose(), newTensorB).Print()
	//elapsed := time.Since(start)
	//sum := elapsed.Milliseconds()
	//
	//fmt.Println(sum)

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
