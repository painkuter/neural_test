package main

import (
	"fmt"
	"time"

	"neural_test/internal/datastruct"
	"neural_test/internal/dbm"
	"neural_test/internal/fn"

	"gonum.org/v1/gonum/mat"
	"neural_test/internal/net"
)

const (
	epochs       = 70000
	learningRate = 0.03
	m            = 3 // first layer
	n            = 3 // hidden layer
	layersCount  = 4 // layers count
	vectorSize   = 3 // vector size
)

type StartNetwork struct {
	FirstLvlWeight  *mat.Dense
	HiddenLvlWeight *mat.Dense
}

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
var db = dbm.DB()

func main() {
	_, err := db.Exec("CREATE TABLE IF NOT EXISTS attempts (`id` INT(10) NOT NULL AUTO_INCREMENT, PRIMARY KEY (`id`))")
	if err != nil {
		panic(err)
	}

	//RunStartNetwork()
	net.RunDanaNetwork()
}


func RunStartNetwork() {
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
			actualResults = append(actualResults, net.Predict(in1))
			correctResults = append(correctResults, elem.ExpectedPredict)
		}

		err := fn.MSE(actualResults, correctResults)
		fmt.Printf("\rProcess: %v%% ERROR: %1.6v", 100*i/epochs, err)
		if err < 0.001 {
			break
		}
	}
	fmt.Println()
	fn.Print(net.FirstLvlWeight)
	fn.Print(net.HiddenLvlWeight)
	fmt.Println("\n=======")

	for i := 0; i < len(train2); i++ {
		in1 := mat.NewDense(m, 1, train2[i].Row)
		actual := net.Predict(in1)
		fmt.Printf("On %v actual predict: %5.5v\t[%v]\t expected: %v \t result: %v\n", train2[i].Row, actual, actual > .5, train2[i].ExpectedPredict > 0.5, (actual > .5) == (train2[i].ExpectedPredict > 0.5))
	}
	fmt.Printf("time %v ms\n", (time.Now().UnixNano()-t)/1000000)
}

func (net StartNetwork) Train(train datastruct.Input) {

	in := mat.NewDense(m, 1, train.Row) // [m*1]

	in1 := mat.NewDense(n, 1, nil)  // [n*1]
	in1.Mul(net.FirstLvlWeight, in) // [n*m] * [m*1] = [n*1]
	out1 := fn.SigmoidMap(in1)      // [n*1]

	in2 := mat.NewDense(1, 1, nil)
	in2.Mul(net.HiddenLvlWeight, out1)
	out2 := fn.SigmoidMap(in2)
	actual := out2.At(0, 0)

	// Back Propagation
	// hidden layer
	err2 := actual - train.ExpectedPredict
	grad2 := actual * (1 - actual)
	// gradient_layer_2 := SigmoidDx(actual_predict)
	dWeight2 := err2 * grad2
	tmpWeights2 := mat.DenseCopyOf(out1.T()) //
	tmpWeights2.Scale((-1)*dWeight2*learningRate, tmpWeights2)
	net.HiddenLvlWeight.Add(net.HiddenLvlWeight, tmpWeights2) // updated weights

	// first layer
	err1 := mat.NewDense(1, n, nil)
	err1.Scale(dWeight2, net.HiddenLvlWeight)
	// gradient_layer_1 := SigmoidMapDx(outputs_1)
	grad1 := fn.SigmoidMapDx(out1)
	dWeight1 := mat.NewDense(1, n, nil)
	dWeight1.MulElem(err1, grad1.T())
	tmpWeights1 := mat.NewDense(m, n, nil) // [m*n]
	tmpWeights1.Mul(in, dWeight1)          // TODO: проверить это [m*1] * [1*n]
	tmpWeights1.Scale((-1)*learningRate, tmpWeights1)
	net.FirstLvlWeight.Add(net.FirstLvlWeight, tmpWeights1.T())
}

func (net StartNetwork) Predict(in *mat.Dense) float64 {
	in1 := mat.NewDense(n, 1, nil)
	in1.Mul(net.FirstLvlWeight, in)
	out1 := fn.SigmoidMap(in1)
	in2 := mat.NewDense(1, 1, nil)
	in2.Mul(net.HiddenLvlWeight, out1)
	return fn.SigmoidMap(in2).At(0, 0)
}

func Init(n, m int) StartNetwork {

	var net StartNetwork
	net.FirstLvlWeight = mat.NewDense(n, m, fn.GetRandWeights(n*m))
	// net.FirstLvlWeight = mat.NewDense(n, m, []float64{0.79, 0.44, 0.43, 0.85, 0.43, 0.29})
	net.HiddenLvlWeight = mat.NewDense(1, n, fn.GetRandWeights(n*1))
	// net.HiddenLvlWeight = mat.NewDense(1, n, []float64{0.5, 0.52})
	return net
}
