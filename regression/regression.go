package regression

import (
	"fmt"
	"math/rand"

	"github.com/dimandms/gomllib/ndarray"
)

type LinearRegressor struct {
	Weights      *ndarray.Vector
	Bias         float64
	HasIntercept bool
}

func NewLinearRegressor(addItercept bool) LinearRegressor {
	return LinearRegressor{HasIntercept: addItercept}
}

func (lr *LinearRegressor) Train(objects *ndarray.Matrix, targets *ndarray.Vector) error {
	numberOfObjects, numberOfFeatures := objects.Shape()
	numberOfTargets := targets.Shape()

	lr.initWeigths(numberOfFeatures)

	learningRate := 0.1
	numberOfIteration := 0
	maxNumberOfIteration := 500

	for numberOfIteration < maxNumberOfIteration {
		numberOfIteration += 1

		grad := mseGradient(trainX, trainY, inititalWeight, inititalBias)

		inititalWeight -= grad[0] * learningRate
		inititalBias -= grad[1] * learningRate

		loss := mse(trainX, trainY, inititalWeight, inititalBias)
		fmt.Printf("iteration [%d] grad: %v, loss: %v w: %v, b: %v \n", numberOfIteration, grad, loss, inititalWeight, inititalBias)

	}

	lr.Weight = inititalWeight
	lr.Bias = inititalBias

	return nil
}

func (lr *LinearRegressor) initWeigths(numberOfFeatures int) {
	if lr.HasIntercept {
		lr.Bias = rand.Float64()
	}

	weights := make([]float64, numberOfFeatures)
	for i := range weights {
		weights[i] = rand.Float64()
	}

	lr.Weights = ndarray.NewVector(weights)
}

func mse(objects *ndarray.Matrix, trueValues, weights *ndarray.Vector, bias float64) float64 {
	numberOfObjects, _ := objects.Shape()
	biasVector := ndarray.NewVectorFrom(bias, numberOfObjects)

	answer, err := objects.DotVector(weights)
	if err != nil {
		return 0.0
	}

	biasedAnswer, err := answer.AddVector(biasVector)
	if err != nil {
		return 0.0
	}

	errorVectorized, err := trueValues.SubVector(biasedAnswer)
	if err != nil {
		return 0.0
	}
	return errorVectorized.Pow(2).Sum() / float64(numberOfObjects)
}

func mseGradient(objects *ndarray.Matrix, trueValues, weights *ndarray.Vector, bias float64) *ndarray.Vector {
	numberOfObjects, _ := objects.Shape()
	biasVector := ndarray.NewVectorFrom(bias, numberOfObjects)

	answer, err := objects.DotVector(weights)
	if err != nil {
		return nil
	}

	biasedAnswer, err := answer.AddVector(biasVector)
	if err != nil {
		return nil
	}

	errorVectorized, err := biasedAnswer.SubVector(trueValues)
	if err != nil {
		return nil
	}

	tempResult, err := objects.Transpose().DotVector(errorVectorized)
	if err != nil {
		return nil
	}

	return tempResult.MultiplicateBy(1 / float64(numberOfObjects))
}
