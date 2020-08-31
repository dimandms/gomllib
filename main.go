package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/dimandms/gomllib/ndarray"
	"github.com/dimandms/gomllib/regression"
)

func main() {
	X := make([][]float64, 0)
	y := make([]float64, 0)

	w1 := 0.5
	w2 := 0.3
	b := 0.1

	numOfPoints := 50
	n := 0

	for n < numOfPoints {
		x1 := float64(n) / 50.0
		x2 := math.Sqrt(x1)
		X = append(X, []float64{x1, x2})
		y = append(y, w1*x1+w2*x2+b+rand.Float64()/20)
		n += 1
	}

	reg := regression.NewLinearRegressor(true)
	err := reg.Train(ndarray.NewMatrix(X), ndarray.NewVector(y))
	if err != nil {
		fmt.Println("oops")
	}
}
