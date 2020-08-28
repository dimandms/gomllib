package regression

import (
	"fmt"
	"math"
	"math/rand"
)

type LinearRegressor struct {
	Weight float64
	Bias   float64
}

func NewLinearRegressor() LinearRegressor {
	return LinearRegressor{}
}

func (lr *LinearRegressor) Train(trainX []float64, trainY []float64) error {
	xLength := len(trainX)
	yLength := len(trainY)

	if xLength != yLength {
		return fmt.Errorf("Regressor train error: X and Y length missmatch: %v, %v", xLength, yLength)
	}

	inititalWeight := rand.Float64()
	inititalBias := rand.Float64()

	learningRate := 0.001
	numberOfIteration := 0
	maxNumberOfIteration := 1000

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

func mse(objects, trueValues []float64, weight, bias float64) float64 {
	total := 0.0

	for i, object := range objects {
		total += math.Pow(trueValues[i]-(object*weight+bias), 2)
	}

	return total / float64(len(trueValues))
}

func mseGradient(objects, trueValues []float64, weight, bias float64) []float64 {
	result := []float64{0.0, 0.0}
	numOfObjects := float64(len(objects))

	for i, object := range objects {
		result[0] += -2 * object * (trueValues[i] - (object*weight + bias))
		result[1] += -2 * (trueValues[i] - (object*weight + bias))
	}

	return []float64{result[0] / numOfObjects, result[1] / numOfObjects}
}
