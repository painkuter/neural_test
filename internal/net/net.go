package net

import (
	"neural_test/internal/datastruct"
	"neural_test/internal/fn"

	"fmt"
	"time"

	"gonum.org/v1/gonum/mat"
)

type DanaNetwork struct {
	W            []*mat.Dense
	VectorSize   int     // Rows count
	LayersCount  int     // cols count
	learningRate float64 // learningRate
}

type DanaResult []*mat.Dense

func (d DanaResult) GetResult() float64 {
	return d[len(d)-1].At(0, 0)
}

func RunDanaNetwork() {
	var train = []datastruct.Input{
		{[]float64{0, 0, 0}, 0},
		{[]float64{0, 0, 1}, 0},
		{[]float64{0, 1, 0}, 0},
		{[]float64{0, 1, 1}, 1},
		{[]float64{1, 0, 0}, 0},
		{[]float64{1, 0, 1}, 1},
		{[]float64{1, 1, 0}, 1},
		//{[]float64{1, 1, 1}, 1},
	}

	var train2 = append(train, datastruct.Input{[]float64{1, 1, 1}, 1})
	var (
		epochs = 70000
		m      = 3 // first layer
		n      = 3 // hidden layer
	)
	net := Init(n, m)
	t := time.Now().UnixNano()

	fmt.Printf("STARTING TRAINING: %v EPOCHS FOR %v INPUTS = %v\n", epochs, len(train), epochs*len(train))
	for i := 0; i < epochs; i++ {
		for _, elem := range train {
			net.Train(elem)
		}
		var actualResults []float64
		var correctResults []float64
		for _, elem := range train {
			in1 := mat.NewDense(m, 1, elem.Row)
			actualResults = append(actualResults, net.Predict(in1).GetResult())
			correctResults = append(correctResults, elem.ExpectedPredict)
		}

		err := fn.MSE(actualResults, correctResults)
		fmt.Printf("\rProcess: %v%% ERROR: %1.6v", 100*i/epochs, err)
		if err < 0.001 {
			break
		}
	}
	fmt.Println()
	fmt.Println("\n=======")

	for i := 0; i < len(train2); i++ {
		in1 := mat.NewDense(m, 1, train2[i].Row)
		actual := net.Predict(in1).GetResult()
		fmt.Printf("On %v actual predict: %5.5v\t[%v]\t expected: %v \t result: %v\n", train2[i].Row, actual, actual > .5, train2[i].ExpectedPredict > 0.5, (actual > .5) == (train2[i].ExpectedPredict > 0.5))
	}
	fmt.Printf("time %v ms\n", (time.Now().UnixNano()-t)/1000000)
}

func Init(vs, lc int) DanaNetwork {
	net := DanaNetwork{
		VectorSize:  vs,
		LayersCount: lc,
	}
	for i := 0; i < net.LayersCount-1; i++ {
		net.W = append(net.W, mat.NewDense(net.VectorSize, net.VectorSize, fn.GetRandWeights(net.VectorSize*net.VectorSize)))
	}

	net.W = append(net.W, mat.NewDense(1, net.VectorSize, fn.GetRandWeights(net.VectorSize)))
	return net
}

func (net *DanaNetwork) Train(train datastruct.Input) {
	in := mat.NewDense(net.VectorSize, 1, train.Row)

	/*out := mat.NewDense(net.VectorSize, 1, nil)
	for i := 0; i < net.LayersCount-1; i++ {
		out.Mul(net.W[i], in) // [vs*vs] * [vs*1] = [vs*1]
		in = fn.SigmoidMap(out)
	}

	lastOne := mat.NewDense(1, 1, nil)
	lastOne.Mul(net.W[net.LayersCount], in)
	result := fn.SigmoidMap(lastOne)
	actual := result.At(0, 0)
	_ = actual*/

	result := net.Predict(in)

	// Back Propagation
	// last layer
	err := result.GetResult() - train.ExpectedPredict
	grad := result.GetResult() * (1 - result.GetResult())
	dWeight := err * grad

	tmpWeights := mat.DenseCopyOf(result[net.VectorSize-2].T())
	tmpWeights.Scale((-1)*dWeight*net.learningRate, tmpWeights)
	fmt.Println(tmpWeights.Dims())
	fmt.Println(net.W[net.VectorSize-1].Dims())
	net.W[net.VectorSize-1].Add(net.W[net.VectorSize-1], tmpWeights)

	// common layer
	for i := 1; i < net.VectorSize; i++ {
		err := mat.NewDense(1, net.VectorSize, nil)
		err.Scale(dWeight, net.W[net.VectorSize-i])

		grad := fn.SigmoidMapDx(result[net.VectorSize-i])
		dWeight := mat.NewDense(1, net.VectorSize, nil)
		dWeight.MulElem(err, grad.T())
		tmpWeights := mat.NewDense(net.VectorSize, net.VectorSize, nil) // [m*n]
		tmpWeights.Mul(in, dWeight)                                     // TODO: check size
		tmpWeights.Scale((-1)*net.learningRate, tmpWeights)
		net.W[net.VectorSize-1-i].Add(net.W[net.VectorSize-1-i], tmpWeights.T())
	}
}

func (net *DanaNetwork) Predict(in *mat.Dense) DanaResult {
	var result DanaResult
	out := mat.NewDense(net.VectorSize, 1, nil)
	for i := 0; i < net.LayersCount-1; i++ {
		out.Mul(net.W[i], in) // [vs*vs] * [vs*1] = [vs*1]
		result = append(result, out)
		in = fn.SigmoidMap(out)
	}

	lastOne := mat.NewDense(1, 1, nil)
	lastOne.Mul(net.W[net.LayersCount-1], in)
	result = append(result, fn.SigmoidMap(lastOne))
	return result
}
