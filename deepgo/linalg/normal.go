package linalg

import (
	"math"
	"math/rand"
)

func NormalDistribution(row, col int) Matrix[float64] {

	// Tamanho desejado da amostra
	size := row * col

	// Semente aleatória
	rand.Seed(0)

	// Criar um slice para armazenar a amostra
	sample := make([]float64, size)

	// Gerar a amostra da distribuição normal
	for i := 0; i < size; i += 2 {
		u1 := rand.Float64()
		u2 := rand.Float64()

		z1 := math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
		z2 := math.Sqrt(-2*math.Log(u1)) * math.Sin(2*math.Pi*u2)

		sample[i] = z1
		if i+1 < size {
			sample[i+1] = z2
		}
	}

	return NewMatrix(sample, row, col)

}
