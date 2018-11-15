package fn

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

func GetRandWeights(size int) []float64 {
	weights := make([]float64, 0, size)
	rand.Seed(time.Now().UnixNano())
	for i := 0; i < size; i++ {
		weights = append(weights, Round(rand.Float64(), .5, 4))
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

func Delta(a, b float64) float64 {
	return math.Sqrt(a*a + b*b)
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
		//out.RawMatrix().Data[i] = SigmoidDx(in.RawMatrix().Data[i])
		out.RawMatrix().Data[i] = in.RawMatrix().Data[i] * (1 - in.RawMatrix().Data[i])
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

func MSE(actual, expected []float64) float64 {
	r := 0.
	for i := range actual {
		r += math.Pow((actual[i] - expected[i]), 2)
	}
	return r / float64(len(actual))
}
