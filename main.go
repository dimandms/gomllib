package main

import (
	"fmt"

	"github.com/dimandms/gomllib/classification"
	"github.com/dimandms/gomllib/datasets"
)

func main() {
	X, y, err := datasets.LoadIris()
	if err != nil {
		fmt.Println(err)
	}

	fmt.Printf("X: %v\n", X[0:4])
	fmt.Printf("y: %v\n", y[0:4])

	clf := classification.New(false)

	err = clf.Train(X[0:98], y[0:98])
	if err != nil {
		fmt.Println("error")
	}

	fmt.Printf("clf.Weights: %v\n", clf.Weights)

	probs, err := clf.PredictProbabilities(X[50:53])
	if err != nil {
		fmt.Println("error")
	}

	fmt.Printf("probs: %v\n", probs)
}
