package main

func (this Network) oldTrain(train Input) {

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
}
