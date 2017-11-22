package main

import (
	"fmt"
	"math"
	"time"

	"gonum.org/v1/gonum/mat"
)

const (
	epochs       = 1
	learningRate = 0.01
	m            = 3 // first layer
	n            = 3 // hidden layer
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
	{[]float64{0, 0, 1}, 0},
	{[]float64{0, 1, 0}, 0},
	{[]float64{0, 1, 1}, 1},
	{[]float64{1, 0, 0}, 0},
	{[]float64{1, 0, 1}, 1},
	{[]float64{1, 1, 0}, 1},
	//{[]float64{1, 1, 1}, 1},
}

var train2 = append(train, Input{[]float64{1, 1, 1}, 1})
var db = DB()

func main() {
	_, err := db.Exec("CREATE TABLE IF NOT EXISTS attempts (`id` INT(10) NOT NULL AUTO_INCREMENT, PRIMARY KEY (`id`))")
	if err != nil {
		panic(err)
	}
	net := Init(n, m)
	t := time.Now().UnixNano()

	fmt.Printf("STARTING TRAINING WITH ON %v EPOCHS FOR %v INPUTS = %v\n", epochs, len(train), epochs*len(train))
	for i := 0; i < epochs; i++ {
		for _, elem := range train {
			net.Train(elem)
		}
		var actual_results []float64
		var correct_results []float64
		for _, elem := range train {
			in1 := mat.NewDense(m, 1, elem.Row)
			actual_results = append(actual_results, net.Predict(in1))
			correct_results = append(correct_results, elem.ExpectedPredict)
		}

		err := MSE(actual_results, correct_results)
		fmt.Printf("\rProcess: %v%% ERROR: %1.6v", 100*i/epochs, err)
		if err < 0.001 {
			break
		}
	}
	fmt.Println()
	Print(net.FirstLvlWeight)
	Print(net.HiddenLvlWeight)
	fmt.Println("\n=======")

	for i := 0; i < len(train2); i++ {
		in1 := mat.NewDense(m, 1, train2[i].Row)
		actual := net.Predict(in1)
		fmt.Printf("On %v actual predict: %5.5v\t[%v]\t expected: %v \t result: %v\n", train2[i].Row, actual, actual > .5, train2[i].ExpectedPredict > 0.5, (actual > .5) == (train2[i].ExpectedPredict > 0.5))
	}
	fmt.Printf("time %v ms\n", (time.Now().UnixNano()-t)/1000000)
}

func (this Network) Train(train Input) {
	////predict
	//	in1 := mat.NewDense(m, 1, train.Row) //[m*1]
	//	out1 := mat.NewDense(n, 1, nil)      //[n*1]
	//	out1.Mul(this.FirstLvlWeight, in1)   //[n*m]*[m*1] = [n*1]?
	//	in2 := SigmoidMap(out1)              //[n*1]
	//
	//	out2 := mat.NewDense(1, 1, nil)
	//	out2.Mul(this.HiddenLvlWeight, in2)
	//	result := SigmoidMap(out2)
	//	actual := result.At(0, 0) // = out2
	//
	//	//get error
	//	err2 := actual - train.ExpectedPredict
	//	gradient2 := actual * (1 - actual) // sigmoid(x)dx|x=actual
	//	dWeight2 := err2 * gradient2
	//
	//	// w2 = w2 - in2 * learningRate * d_w : [1*n] = [1*n] - [n*1]^T * [1] * [1]
	//	tmpWeights2 := mat.DenseCopyOf(in2.T()) //
	//	tmpWeights2.Scale((-1)*dWeight2*learningRate, tmpWeights2)
	//	this.HiddenLvlWeight.Add(this.HiddenLvlWeight, tmpWeights2) //updated weights
	//
	//	//обратное распространение
	//
	//	err1 := mat.NewDense(1, n, nil)            // [1*n]
	//	err1.Scale(dWeight2, this.HiddenLvlWeight) //[1*n]
	//	gradient1 := SigmoidMapDx(in2).T()             //[n*1]
	//	//Print(gradient1)
	//	dWeight1 := mat.NewDense(1, n, nil)   //[1*n]
	//	dWeight1.MulElem(err1, gradient1) //[1*n] ** [1*n] Hadamard product
	//
	//	tmpWeights1 := mat.NewDense(m, n, nil) // [m*n]
	//	tmpWeights1.Mul(in1, dWeight1) // [m*1] * [1*n]
	//	tmpWeights1.Scale((-1)*learningRate, tmpWeights1)
	//	this.FirstLvlWeight.Add(this.FirstLvlWeight, tmpWeights1.T())
	//	return

	//inputs_1 = np.dot(self.weights_0_1, inputs)
	in := mat.NewDense(m, 1, train.Row) // [m*1]
	in1 := mat.NewDense(n, 1, nil)      // [n*1]
	in1.Mul(this.FirstLvlWeight, in)    // [n*m] * [m*1] = [n*1]
	out1 := SigmoidMap(in1)
	in2 := mat.NewDense(1, 1, nil)
	in2.Mul(this.HiddenLvlWeight, out1)
	out2 := SigmoidMap(in2)
	actual := out2.At(0, 0)

	//BackPrepagation
	//hidden layer
	err2 := actual - train.ExpectedPredict
	grad2 := actual * (1 - actual)
	//gradient_layer_2 := SigmoidDx(actual_predict)
	dWeight2 := err2 * grad2
	tmpWeights2 := mat.DenseCopyOf(out1.T()) //
	tmpWeights2.Scale((-1)*dWeight2*learningRate, tmpWeights2)
	this.HiddenLvlWeight.Add(this.HiddenLvlWeight, tmpWeights2) //updated weights

	//first layer
	err1 := mat.NewDense(1, n, nil)
	err1.Scale(dWeight2, this.HiddenLvlWeight)
	//gradient_layer_1 := SigmoidMapDx(outputs_1)
	grad1 := SigmoidMapDx(out1)
	dWeight1 := mat.NewDense(1, n, nil)
	dWeight1.MulElem(err1, grad1.T())
	tmpWeights1 := mat.NewDense(m, n, nil) // [m*n]
	tmpWeights1.Mul(in, dWeight1)          //TODO: проверить это [m*1] * [1*n]
	tmpWeights1.Scale((-1)*learningRate, tmpWeights1)
	this.FirstLvlWeight.Add(this.FirstLvlWeight, tmpWeights1.T())
}

func (this Network) Predict(in *mat.Dense) float64 {
	in1 := mat.NewDense(n, 1, nil)
	in1.Mul(this.FirstLvlWeight, in)
	out1 := SigmoidMap(in1)
	in2 := mat.NewDense(1, 1, nil)
	in2.Mul(this.HiddenLvlWeight, out1)
	return SigmoidMap(in2).At(0, 0)
}

func Init(n, m int) Network {
	var net Network
	net.FirstLvlWeight = mat.NewDense(n, m, GetRandWeights(n*m))
	//net.FirstLvlWeight = mat.NewDense(n, m, []float64{0.79, 0.44, 0.43, 0.85, 0.43, 0.29})
	net.HiddenLvlWeight = mat.NewDense(1, n, GetRandWeights(n*1))
	//net.HiddenLvlWeight = mat.NewDense(1, n, []float64{0.5, 0.52})
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
