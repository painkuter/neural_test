package main

import (
	"fmt"
	//dense"github.com/mitsuse/matrix-go"
	"github.com/mitsuse/matrix-go/dense"
)

const (
	learningRate = 0.1
)

func main() {
	fmt.Println("Test")
	m := dense.New(2, 3)(
		0, 1, 2,
		3, 4, 5,
	)
	fmt.Println(m)
}

type Network struct {
	FirstLvlWeight  [][]float64
	SecondLvlWeight []float64
}
