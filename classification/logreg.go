package logreg

import "fmt"

type LogisticRegression struct {
	trainX [][]float64
	trainY []float64
}

func New() LogisticRegression {
	return LogisticRegression{}
}

func (lr LogisticRegression) fit() error {
	return fmt.Errorf("Not implemented")
}
