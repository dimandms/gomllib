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
	objects = lr.preprocess(objects)
	_, n := objects.Shape()
	lr.initWeigths(n)

	learningRate := 0.01

	for i := 0; i < 500; i++ {
		err := lr.step(objects, targets, learningRate)
		if err != nil {
			return err
		}

		fmt.Printf("iteration [%d] loss: %v w: %v\n", i, mse(objects, targets, lr.Weights), lr.Weights)
	}

	return nil
}

func (lr *LinearRegressor) preprocess(objects *ndarray.Matrix) *ndarray.Matrix {
	n, _ := objects.Shape()

	if lr.HasIntercept {
		fakeBiasFeature := ndarray.NewVectorFrom(1.0, n)
		return objects.ExtendWith(fakeBiasFeature)
	}
	return objects
}

func (lr *LinearRegressor) step(objects *ndarray.Matrix, targets *ndarray.Vector, learningRate float64) error {
	grad := mseGradient(objects, targets, lr.Weights)

	newWeights, err := lr.Weights.SubVector(grad.MultiplicateBy(learningRate))
	if err != nil {
		return err
	}
	lr.Weights = newWeights

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
