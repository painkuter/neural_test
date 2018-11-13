package net

import (
	"neural_test/internal/fn"

	"gonum.org/v1/gonum/mat"
)

type Network2 []*mat.Dense

func Init2(vs, lc int) Network2 {
	var net Network2
	for i := 0; i < lc; i++ {
		net = append(net, mat.NewDense(vs, vs, fn.GetRandWeights(vs^2)))
	}
	return net
}
