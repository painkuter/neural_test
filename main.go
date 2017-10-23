package main

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

const (
	learningRate = 0.1
	m            = 3 // first layer
	n            = 2 // hidden layer
)

type Input struct {
	Row             []float64
	ExpectedPredict float64
}

var train = []Input{
	{[]float64{0, 0, 0}, 0},
	{[]float64{0, 0, 1}, 1},
	{[]float64{0, 1, 0}, 0},
	{[]float64{0, 1, 1}, 0},
	{[]float64{1, 0, 0}, 1},
	{[]float64{1, 0, 1}, 1},
	{[]float64{1, 1, 0}, 0},
	{[]float64{1, 1, 1}, 0},
}

func main() {
	net := Init(m, n)

	in := mat.NewDense(m, 1, train[0].Row)
	net.Predict(in)
}

type Network struct {
	FirstLvlWeight  *mat.Dense
	HiddenLvlWeight *mat.Dense
}

func Init(m, n int) Network {
	var net Network
	net.FirstLvlWeight = mat.NewDense(n, m, GetRandWeights(n*m))
	net.HiddenLvlWeight = mat.NewDense(1, n, GetRandWeights(n*1))
	return net
}

func (this Network) Train(in, actualPredict, expectedPredict float64) {
	errLayer2 := actualPredict - expectedPredict
	gradient2 := actualPredict * (1 - actualPredict)
	weightDelta2 := mat.NewDense(this.FirstLvlWeight.RawMatrix().Rows, 1, nil)

	this.HiddenLvlWeight
	return
}

func (this Network) Predict(in *mat.Dense) {
	in1 := mat.NewDense(this.FirstLvlWeight.RawMatrix().Rows, 1, nil)
	in1.Mul(this.FirstLvlWeight, in)
	out1 := SigmoidMap(in1)

	//Print(this.FirstLvlWeight)
	//Print(in)
	//Print(out1)
	//Print(this.HiddenLvlWeight)

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
