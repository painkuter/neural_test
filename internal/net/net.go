package net

import (
	"neural_test/internal/datastruct"
	"neural_test/internal/fn"

	"gonum.org/v1/gonum/mat"
)

type DanaNetwork struct {
	W           []*mat.Dense
	VectorSize  int // Rows count
	LayersCount int // cols count
}

type DanaResult []*mat.Dense

func Init(vs, lc int) DanaNetwork {
	net := DanaNetwork{
		VectorSize:  vs,
		LayersCount: lc,
	}
	for i := 0; i < net.LayersCount-1; i++ {
		net.W = append(net.W, mat.NewDense(net.VectorSize, net.VectorSize, fn.GetRandWeights(net.VectorSize^2)))
	}

	net.W = append(net.W, mat.NewDense(1, net.VectorSize, fn.GetRandWeights(net.VectorSize^2)))
	return net
}

func (net DanaNetwork) Train(train datastruct.Input) {
	in := mat.NewDense(net.VectorSize, 1, train.Row)

	out := mat.NewDense(net.VectorSize, 1, nil)
	for i := 0; i < net.LayersCount-1; i++ {
		out.Mul(net.W[i], in) // [vs*vs] * [vs*1] = [vs*1]
		in = fn.SigmoidMap(out)
	}

	lastOne := mat.NewDense(1, 1, nil)
	lastOne.Mul(net.W[net.LayersCount], in)
	result := fn.SigmoidMap(lastOne)
	actual := result.At(0, 0)
	_ = actual
}

func (net DanaNetwork) Predict(in *mat.Dense) DanaResult {

	out := mat.NewDense(net.VectorSize, 1, nil)
	for i := 0; i < net.LayersCount-1; i++ {
		out.Mul(net.W[i], in) // [vs*vs] * [vs*1] = [vs*1]
		in = fn.SigmoidMap(out)
	}

	lastOne := mat.NewDense(1, 1, nil)
	lastOne.Mul(net.W[net.LayersCount], in)
	result := fn.SigmoidMap(lastOne)
	actual := result.At(0, 0)
}
