package preprocessing

import "tensors-processing/deepgo/linalg"

func OneHotEncoder[T linalg.NumTypes](y []T) linalg.Matrix[T] {
	// Find the number of categories
	numCategories := T(0)
	for _, val := range y {
		if val+T(1) > numCategories {
			numCategories = val + 1
		}
	}

	// Prepare a slice to hold the one-hot representation
	oneHot := make([][]T, len(y))
	for i := range oneHot {
		oneHot[i] = make([]T, int(numCategories))
	}

	// Convert to one-hot
	for i, val := range y {
		oneHot[i][int(val)] = 1
	}

	return linalg.NewMatrixFrom2D(oneHot, len(y), int(numCategories))
}
