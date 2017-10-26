package main

import (
	"math"

	"fmt"
	"gonum.org/v1/gonum/mat"
)

const (
	epochs       = 500
	learningRate = 0.1
	m            = 3 // first layer
	n            = 2 // hidden layer
)

type Input struct {
	Row             []float64
	ExpectedPredict float64
}

type Network struct {
	FirstLvlWeight  *mat.Dense
	HiddenLvlWeight *mat.Dense
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

	//in1 := mat.NewDense(m, 1, train[0].Row)
	//actual := net.Predict(in1)
	//
	//net.Train(actual.At(0, 0), train[0].ExpectedPredict)
	for i := 0; i < epochs; i++ {
		for _, elem := range train {
			net.Train(elem)
		}
	}
	fmt.Println("=======")
	in1 := mat.NewDense(m, 1, train[0].Row)
	actual := net.Predict(in1)
	Print(actual)
}

func (this Network) Train(train Input) {
	in1 := mat.NewDense(m, 1, train.Row)
	out1 := mat.NewDense(this.FirstLvlWeight.RawMatrix().Rows, 1, nil)
	out1.Mul(this.FirstLvlWeight, in1)
	in2 := SigmoidMap(out1)

	out2 := mat.NewDense(1, 1, nil)
	out2.Mul(this.HiddenLvlWeight, in2)
	result := SigmoidMap(out2)
	actual := result.At(0, 0)

	err2 := actual - train.ExpectedPredict
	gradient2 := actual * (1 - actual)
	dWeight2 := err2 * gradient2

	tmpWeights2 := mat.DenseCopyOf(this.HiddenLvlWeight)
	tmpWeights2.Scale((-1)*dWeight2, tmpWeights2)
	this.HiddenLvlWeight.Add(this.HiddenLvlWeight, tmpWeights2) //updated weights

	err1 := mat.NewDense(1, n, nil)
	err1.Scale(dWeight2, this.HiddenLvlWeight)
	gradient1 := SigmoidMapDx(in2)
	dWeight1 := mat.NewDense(1, n, nil)
	dWeight1.Mul(err1, gradient1)
	Print(dWeight1)

	//tmpWeights1 := mat.DenseCopyOf(this.FirstLvlWeight)
	//tmpWeights1.Mul(in1,)

	//
	//weights := mat.NewDense(this.FirstLvlWeight.RawMatrix().Rows, 1, nil)
	//weights.Mul(weightDelta2, actualPredict)

	Print(this.HiddenLvlWeight)
	return
}

func (this Network) Predict(in1 *mat.Dense) *mat.Dense {
	out1 := mat.NewDense(this.FirstLvlWeight.RawMatrix().Rows, 1, nil)
	out1.Mul(this.FirstLvlWeight, in1)
	in2 := SigmoidMap(out1)

	out2 := mat.NewDense(1, 1, nil)
	out2.Mul(this.HiddenLvlWeight, in2)
	actual := SigmoidMap(out2)
	//Print(actual)
	return actual
}

func Init(m, n int) Network {
	var net Network
	net.FirstLvlWeight = mat.NewDense(n, m, GetRandWeights(n*m))
	net.HiddenLvlWeight = mat.NewDense(1, n, GetRandWeights(n*1))
	return net
}

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func SigmoidDx(x float64) float64 {
	return Sigmoid(x) * (1 - Sigmoid(x))
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

func SigmoidMapDx(in *mat.Dense) *mat.Dense {
	if in.RawMatrix().Cols != 1 {
		panic("Wrong dimension")
	}
	out := mat.NewDense(in.RawMatrix().Rows, 1, nil)
	for i := 0; i < in.RawMatrix().Rows; i++ {
		out.RawMatrix().Data[i] = SigmoidDx(in.RawMatrix().Data[i])
	}
	return out
}

// E returns identity column by size
func E(size int) *mat.Dense {
	vector := make([]float64, 0, size)
	for i := 0; i < size; i++ {
		vector = append(vector, 1)
	}
	e := mat.NewDense(size, 1, vector)
	return e
}
