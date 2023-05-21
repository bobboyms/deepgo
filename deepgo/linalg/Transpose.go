package linalg

func Transpose[T NumTypes](data []T, row, col int) []T {

	tempMtx := GetRow(data, row, col)

	var newData []T
	for i := 0; i < col; i++ {
		for y := 0; y < row; y++ {
			newData = append(newData, tempMtx[y][i])
		}
	}
	return newData
}
