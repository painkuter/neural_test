package main

import (
	"testing"

	"neural_test/internal/fn"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

//go test -run GetRandWeights
func TestGetRandWeights(t *testing.T) {
	assert.Equal(t, len(fn.GetRandWeights(6)), 6)
	//fmt.Println(GetRandWeights(20))
}

func TestRound(t *testing.T) {
	assert.Equal(t, fn.Round(0.12345, .5, 2), 0.12, "Round")
}

func TestE(t *testing.T) {
	expectedResult := mat.NewDense(3, 1, []float64{1, 1, 1})
	assert.ObjectsAreEqual(fn.E(3), expectedResult)
}

func Test(t *testing.T) {
	main()
}

/*
func TestSum (t *testing.T){
	a := mat.NewDense(2, 2, []float64{
		1,0,
		2,1,
	})
	b := mat.NewDense(2, 2, []float64{
		1,0,
		0,3,
	})
	Print(a)
	Print(b)
	b.Add(a.T(),b)
	Print(a)
	Print(b)
}
*/

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
