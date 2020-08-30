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
	_, numberOfFeatures := objects.Shape()

	lr.initWeigths(numberOfFeatures)

	learningRate := 0.01
	numberOfIteration := 0
	maxNumberOfIteration := 500

	for numberOfIteration < maxNumberOfIteration {
		numberOfIteration += 1

		grad, biasDerivative := mseGradient(objects, targets, lr.Weights, lr.Bias)

		updatedWeights, err := lr.Weights.SubVector(grad.MultiplicateBy(learningRate))
		if err != nil {
			return err
		}
		lr.Weights = updatedWeights

		lr.Bias -= biasDerivative * learningRate

		loss := mse(objects, targets, lr.Weights, lr.Bias)
		fmt.Printf("iteration [%d] grad: %v, loss: %v w: %v, b: %v \n", numberOfIteration, grad, loss, lr.Weights, lr.Bias)

	}

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

func mseGradient(objects *ndarray.Matrix, trueValues, weights *ndarray.Vector, bias float64) (*ndarray.Vector, float64) {
	numberOfObjects, _ := objects.Shape()
	biasVector := ndarray.NewVectorFrom(bias, numberOfObjects)

	answer, err := objects.DotVector(weights)
	if err != nil {
		return nil, 0.0
	}

	biasedAnswer, err := answer.AddVector(biasVector)
	if err != nil {
		return nil, 0.0
	}

	errorVectorized, err := biasedAnswer.SubVector(trueValues)
	if err != nil {
		return nil, 0.0
	}

	tempResult, err := objects.Transpose().DotVector(errorVectorized)
	if err != nil {
		return nil, 0.0
	}

	partialBiasDerivative := errorVectorized.Sum()

	return tempResult.MultiplicateBy(1 / float64(numberOfObjects)), partialBiasDerivative
}
