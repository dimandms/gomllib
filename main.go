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

	numOfPoints := 50
	n := 0

	for n < numOfPoints {
		x1 := float64(n)
		x2 := math.Sqrt(x1)
		X = append(X, []float64{x1, x2})
		y = append(y, exampleFunc(x1, x2)+rand.Float64()/20)
		n += 1
	}

	reg := regression.NewLinearRegressor(true)

	scaler := ndarray.NewStandardScaler()
	data := scaler.FitTransform(ndarray.NewMatrix(X))
	fmt.Println(data)
	err := reg.Train(data, ndarray.NewVector(y))
	if err != nil {
		fmt.Println("oops")
	}

	trueValue := exampleFunc(25., 15.0)
	preds := reg.Predict(ndarray.NewMatrix([][]float64{{25.0, 15.0}}))

	fmt.Printf("true: %v preds: %v", trueValue, preds)
}

func exampleFunc(x1, x2 float64) float64 {
	w1 := 0.5
	w2 := 0.3
	b := 0.1

	return w1*x1 + w2*x2 + b
}
