package linalg

func Mul[T NumTypes](matrixA, matrixB Matrix[T]) Matrix[T] {
	return ProcessOperation(matrixA, matrixB, MulVec[T])
}

func Div[T NumTypes](matrixA, matrixB Matrix[T]) Matrix[T] {
	return ProcessOperation(matrixA, matrixB, DivVec[T])
}

func Sub[T NumTypes](matrixA, matrixB Matrix[T]) Matrix[T] {
	return ProcessOperation(matrixA, matrixB, SubVec[T])
}

func Sum[T NumTypes](matrixA, matrixB Matrix[T]) Matrix[T] {
	return ProcessOperation(matrixA, matrixB, SumVec[T])
}
