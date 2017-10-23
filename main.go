package main

import (
	"fmt"
	//mat"github.com/mitsuse/matrix-go"
	"math"

	"github.com/mitsuse/matrix-go/dense"
	"gonum.org/v1/gonum/mat"
)

const (
	learningRate = 0.1
	m            = 3 // first layer
	n            = 2 // hidden layer
)

func main() {
	a := mat.NewDense(n, n, []float64{
		.11, .12,
		.21, .22,
	})
	Print(a)

	b := mat.NewDense(m, n, []float64{
		.41, .42,
		.51, .52,
		.61, .62,
	})
	Print(b)

	c := mat.NewDense(m, n, nil)

	c.Mul(b, a)
	Print(c)
}

type Network struct {
	FirstLvlWeight *dense.Matrix
	//FirstLvlWeight  [][]float64
	HiddenLvlWeight *dense.Matrix
	//HiddenLvlWeight []float64
}

func Init(m, n int) Network {
	var net Network
	net.FirstLvlWeight = dense.New(n, m)(
		GetRandWeights(n * m)...,
	)
	net.HiddenLvlWeight = dense.New(n, m)(
		GetRandWeights(n * m)...,
	)
	return net
}

func (this Network) Predict(in *dense.Matrix) {
	out1 := this.FirstLvlWeight.Multiply(in)
	fmt.Println(out1)
}

func Sigmoid(x float64) float64 {
	return 1 / (1 + 1/math.Exp(-x))
}

func Print(m *mat.Dense) {
	for i := 0; i < m.RawMatrix().Rows; i++ {
		for j := 0; j < m.RawMatrix().Cols; j++ {
			fmt.Printf("%6f\t", m.RawMatrix().Data[i*m.RawMatrix().Cols+j])
		}
		fmt.Println()
	}
	fmt.Println()
}
