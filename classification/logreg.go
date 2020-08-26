package logreg

import (
	"fmt"
	"math"
	"math/rand"
)

type LogisticRegression struct {
	weights       []float64
	fit_intercept bool
}

func New(fit_intercept bool) *LogisticRegression {
	return &LogisticRegression{fit_intercept: fit_intercept}
}

func (lr LogisticRegression) Train(trainX [][]float64, trainY []float64) error {
	xLength := len(trainX)
	yLength := len(trainY)
	weightsLength := len(trainX[0])

	if xLength != yLength {
		return fmt.Errorf("Logreg train error: X and Y length missmatch: %v, %v", xLength, yLength)
	}

	inititalWeights := make([]float64, weightsLength)
	for i, _ := range inititalWeights {
		inititalWeights[i] = rand.Float64()
	}

	learningRate := 0.1
	epsilon := 0.001
	initialEpsilon := 9999.0

	for initialEpsilon > epsilon {
		totalGradient := make([]float64, weightsLength)

		for i, object := range trainX {
			partialGradient, err := logLossGradient(object, inititalWeights, trainY[i])
			if err != nil {
				return fmt.Errorf("Logreg train error: %v", err)
			}

			for i := range totalGradient {
				totalGradient[i] += partialGradient[i]
			}
		}

		for i := range totalGradient {
			totalGradient[i] *= learningRate
		}

		for i := range inititalWeights {
			inititalWeights[i] -= totalGradient[i]
		}

		initialEpsilon = totalGradient[0]
		for _, component := range totalGradient {
			initialEpsilon = math.Max(component, initialEpsilon)
		}

	}

	lr.weights = inititalWeights
	return nil
}

func (lr LogisticRegression) PredictProbabilities(objects [][]float64) ([]float64, error) {
	result := make([]float64, 0)

	for _, object := range objects {
		p, err := lr.predictProbability(object)
		if err != nil {
			return nil, fmt.Errorf("Prediction error: %v", err)
		}

		result = append(result, p)
	}

	return result, nil
}

func (lr LogisticRegression) predictProbability(object []float64) (float64, error) {

	multiplied, err := scalarMultiplication(lr.weights, object)
	if err != nil {
		return 0.0, fmt.Errorf("Prediction error: %v", err)
	}

	return sigmoid(multiplied), nil
}

func sigmoid(x float64) float64 {
	return 1.0 / (1 + math.Exp(-x))
}

func logLossGradient(object []float64, weights []float64, trueAnswer float64) ([]float64, error) {
	result := make([]float64, len(weights))

	multiplied, err := scalarMultiplication(weights, object)
	if err != nil {
		return nil, fmt.Errorf("logLossGradient error: %v", err)
	}

	for _, w := range weights {
		result = append(result, w*(sigmoid(multiplied)-trueAnswer))
	}

	return result, nil
}

func logLoss(weights []float64, objects [][]float64, trueAnswers []float64) (float64, error) {
	var result float64

	for i, obj := range objects {
		multiplied, err := scalarMultiplication(weights, obj)
		if err != nil {
			return 0.0, fmt.Errorf("logLoss calcualtion error: %v", err)
		}
		predicted := sigmoid(multiplied)

		result += -1*trueAnswers[i]*math.Log(predicted) - (1.0 - predicted)
	}

	return result / float64(len(objects)), nil
}

func MeanSquareError(weights []float64, objects [][]float64, trueAnswers []float64) (float64, error) {
	var result float64

	for i, obj := range objects {
		answer, err := scalarMultiplication(weights, obj)
		if err != nil {
			return 0.0, fmt.Errorf("MSE calcualtion error: %v", err)
		}

		result += squareError(answer, trueAnswers[i])
	}

	return result, nil

}

func squareError(calculatedAnswer, trueAnswer float64) float64 {
	return math.Pow(calculatedAnswer-trueAnswer, 2)
}

func scalarMultiplication(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0.0, fmt.Errorf("multiplicated vectors should have the same length")
	}

	var result float64
	for i, item := range a {
		result += item * b[i]
	}

	return result, nil
}
