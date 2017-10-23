package main

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestGetRandWeights(t *testing.T) {
	assert.Equal(t, len(GetRandWeights(6)), 6)
}

func TestRound(t *testing.T) {
	assert.Equal(t, Round(0.12345, .5, 2), 0.12, "Round")
}

func Test(t *testing.T) {
	main()
}

/*a := mat.NewDense(m, n, []float64{
	.41, .42,
	.51, .52,
	.61, .62,
})
Print(a)

b := mat.NewDense(n, 1, []float64{
	.11,// .12,
	.21,// .22,
})
Print(b)

c := mat.NewDense(m, 1, nil)

c.Mul(a, b)
Print(c)*/
