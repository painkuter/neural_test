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

func Init(vs, lc int) DanaNetwork {
	net := DanaNetwork{
		VectorSize:  vs,
		LayersCount: lc,
	}
	for i := 0; i < net.LayersCount; i++ {
		net.W = append(net.W, mat.NewDense(net.VectorSize, net.VectorSize, fn.GetRandWeights(net.VectorSize^2)))
	}
	return net
}

func (net DanaNetwork) Train(train datastruct.Input) {
	in := mat.NewDense(net.VectorSize, 1, train.Row)
	_ = in

	out := mat.NewDense(net.VectorSize, 1, nil)
	for i := 0; i < net.LayersCount; i++ {
		out.Mul(net.W[i], in)
		in.Clone(out)
	}
}
