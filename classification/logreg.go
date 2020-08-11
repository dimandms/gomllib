package logreg

import (
	"fmt"
	"math"
)

type LogisticRegression struct {
	trainX        [][]float64
	trainY        []float64
	fit_intercept bool
}

func New(fit_intercept bool) *LogisticRegression {
	return &LogisticRegression{fit_intercept: fit_intercept}
}

func (lr LogisticRegression) Fit(trainX, trainY float64) error {
	return fmt.Errorf("Not implemented")
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
