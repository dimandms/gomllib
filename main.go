package main

import (
	"fmt"
	"math/rand"

	"github.com/dimandms/gomllib/regression"
)

func main() {
	X := make([]float64, 0)
	y := make([]float64, 0)

	w := 0.5
	b := 0.1

	numOfPoints := 50
	n := 0

	for n < numOfPoints {
		x := float64(n) / 50.0
		X = append(X, x)
		y = append(y, w*x+b+rand.Float64()/20)
		n += 1
	}

	fmt.Printf("X: %v \n", X)
	fmt.Printf("y: %v \n", y)

	reg := regression.NewLinearRegressor()
	reg.Train(X, y)
}
