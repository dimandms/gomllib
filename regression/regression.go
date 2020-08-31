package regression

import (
	"fmt"
	"math/rand"

	"github.com/dimandms/gomllib/ndarray"
)

type LinearRegressor struct {
	Weights      *ndarray.Vector
	HasIntercept bool
}

func NewLinearRegressor(addItercept bool) LinearRegressor {
	return LinearRegressor{HasIntercept: addItercept}
}

func (lr *LinearRegressor) Train(objects *ndarray.Matrix, targets *ndarray.Vector) error {
	numberOfObjects, numberOfFeatures := objects.Shape()

	if lr.HasIntercept {
		lr.initWeigths(numberOfFeatures + 1)
		objects = objects.ExtendWith(ndarray.NewVectorFrom(1.0, numberOfObjects))
	} else {
		lr.initWeigths(numberOfFeatures)
	}

	learningRate := 0.01
	numberOfIteration := 0
	maxNumberOfIteration := 500

	for numberOfIteration < maxNumberOfIteration {
		numberOfIteration += 1

		grad := mseGradient(objects, targets, lr.Weights)

		updatedWeights, err := lr.Weights.SubVector(grad.MultiplicateBy(learningRate))
		if err != nil {
			return err
		}
		lr.Weights = updatedWeights

		loss := mse(objects, targets, lr.Weights)
		fmt.Printf("iteration [%d] grad: %v, loss: %v w: %v\n", numberOfIteration, grad, loss, lr.Weights)

	}

	return nil
}

func (lr *LinearRegressor) initWeigths(numberOfFeatures int) {
	weights := make([]float64, numberOfFeatures)
	for i := range weights {
		weights[i] = rand.Float64()
	}

	lr.Weights = ndarray.NewVector(weights)
}

func mse(objects *ndarray.Matrix, trueValues, weights *ndarray.Vector) float64 {
	numberOfObjects, _ := objects.Shape()

	answer, err := objects.DotVector(weights)
	if err != nil {
		return 0.0
	}

	errorVectorized, err := trueValues.SubVector(answer)
	if err != nil {
		return 0.0
	}

	return errorVectorized.Pow(2).Sum() / float64(numberOfObjects)
}

func mseGradient(objects *ndarray.Matrix, trueValues, weights *ndarray.Vector) *ndarray.Vector {
	numberOfObjects, _ := objects.Shape()

	answer, err := objects.DotVector(weights)
	if err != nil {
		return nil
	}

	errorVectorized, err := answer.SubVector(trueValues)
	if err != nil {
		return nil
	}

	tempResult, err := objects.Transpose().DotVector(errorVectorized)
	if err != nil {
		return nil
	}

	return tempResult.MultiplicateBy(1 / float64(numberOfObjects))
}
