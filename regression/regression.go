package regression

import (
	"fmt"
	"math/rand"

	"github.com/dimandms/gomllib/ndarray"
)

const maxNumberOfIteration = 1500
const learningRate = 0.01
const epsilon = 0.00001

//LinearRegressor - type represent linear regression
type LinearRegressor struct {
	Weights      *ndarray.Vector
	HasIntercept bool
}

//NewLinearRegressor is basic constructor LinearRegressor
func NewLinearRegressor(addItercept bool) LinearRegressor {
	return LinearRegressor{HasIntercept: addItercept}
}

//Train is method to train (fit) linear regression model
func (lr *LinearRegressor) Train(objects *ndarray.Matrix, targets *ndarray.Vector) error {
	objects = lr.preprocess(objects)
	_, n := objects.Shape()
	lr.initWeigths(n)

	for i := 0; i < maxNumberOfIteration; i++ {
		previousWeigths := lr.Weights.Copy()
		err := lr.step(objects, targets, learningRate)
		if err != nil {
			return err
		}

		fmt.Printf("iteration [%d] loss: %v w: %v\n", i, mse(objects, targets, lr.Weights), lr.Weights)

		if lr.checkStopIterations(lr.Weights, previousWeigths) {
			break
		}
	}

	return nil
}

//Predict is method to predict answer with trained linear regression model
func (lr *LinearRegressor) Predict(objects *ndarray.Matrix) (*ndarray.Vector, error) {
	objects = lr.preprocess(objects)

	answer, err := objects.DotVector(lr.Weights)
	if err != nil {
		return nil, fmt.Errorf("Linear regressor Predict call failed: %v", err)
	}

	return answer, nil
}

func (lr *LinearRegressor) checkStopIterations(newWeights, previousWeigths *ndarray.Vector) bool {
	delta, err := previousWeigths.SubVector(newWeights)
	if err != nil {
		return false
	}

	maxDelta := delta.AbsMax()
	fmt.Printf("max delta of w: %v \n", maxDelta)

	return maxDelta < epsilon
}

func (lr *LinearRegressor) preprocess(objects *ndarray.Matrix) *ndarray.Matrix {
	n, _ := objects.Shape()

	if lr.HasIntercept {
		fakeBiasFeature := ndarray.NewVectorFrom(1.0, uint(n))
		extended, err := objects.ExtendWith(fakeBiasFeature)
		if err != nil {
			return nil
		}
		return extended
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
