package main

import (
	"fmt"
	//mat"github.com/mitsuse/matrix-go"
	"math"

	//"github.com/mitsuse/matrix-go/dense"
	"gonum.org/v1/gonum/mat"
)

const (
	learningRate = 0.1
	m            = 3 // first layer
	n            = 2 // hidden layer
)

var train = [][]float64{{0, 0, 1}}

func main() {
	net := Init(m, n)

	in := mat.NewDense(m, 1, train[0])
	net.Predict(in)
}

type Network struct {
	FirstLvlWeight *mat.Dense
	//FirstLvlWeight  [][]float64
	HiddenLvlWeight *mat.Dense
	//HiddenLvlWeight []float64
}

func Init(m, n int) Network {
	var net Network
	net.FirstLvlWeight = mat.NewDense(n, m, GetRandWeights(n*m))
	net.HiddenLvlWeight = mat.NewDense(1, n, GetRandWeights(n*1))
	return net
}

func (this Network) Predict(in *mat.Dense) {
	in1 := mat.NewDense(this.FirstLvlWeight.RawMatrix().Rows, 1, nil)
	in1.Mul(this.FirstLvlWeight, in)
	out1 := SigmoidMap(in1)

	Print(this.FirstLvlWeight)
	Print(in)
	Print(out1)
	Print(this.HiddenLvlWeight)

	in2 := mat.NewDense(1, 1, nil)
	in2.Mul(this.HiddenLvlWeight, out1)
	out2 := SigmoidMap(in2)
	Print(out2)
}

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func SigmoidMap(in *mat.Dense) *mat.Dense {
	if in.RawMatrix().Cols != 1 {
		panic("Wrong dimension")
	}
	out := mat.NewDense(in.RawMatrix().Rows, 1, nil)
	for i := 0; i < in.RawMatrix().Rows; i++ {
		out.RawMatrix().Data[i] = Sigmoid(in.RawMatrix().Data[i])
	}
	return out
}

func Print(in *mat.Dense) {
	fmt.Printf("Rows: %v, Cols: %v \n", in.RawMatrix().Rows, in.RawMatrix().Cols)
	for i := 0; i < in.RawMatrix().Rows; i++ {
		for j := 0; j < in.RawMatrix().Cols; j++ {
			fmt.Printf("%3f\t", in.RawMatrix().Data[i*in.RawMatrix().Cols+j])
		}
		fmt.Println()
	}
}
