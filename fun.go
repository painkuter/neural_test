package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

func GetRandWeights(size int) []float64 {
	weights := make([]float64, 0, size)
	for i := 0; i < size; i++ {
		now := time.Now().UnixNano()
		source := rand.NewSource(now)
		randomizer := rand.New(source)
		weights = append(weights, Round(randomizer.Float64(), .5, 2))
	}
	return weights
}

func Round(val float64, roundOn float64, places int) (newVal float64) {
	var round float64
	pow := math.Pow(10, float64(places))
	digit := pow * val
	_, div := math.Modf(digit)
	if div >= roundOn {
		round = math.Ceil(digit)
	} else {
		round = math.Floor(digit)
	}
	newVal = round / pow
	return
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
