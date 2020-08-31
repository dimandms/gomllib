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
	n := 1

	for n < numOfPoints {
		x1 := float64(n)
		x2 := math.Log(x1)
		X = append(X, []float64{x1, x2})
		y = append(y, exampleFunc(x1, x2)+rand.Float64()/20)
		n += 1
	}

	reg := regression.NewLinearRegressor(true)

	scaler := ndarray.NewStandardScaler()
	data := scaler.FitTransform(ndarray.NewMatrix(X))
	targets := ndarray.NewVector(y).ScaleStandard()
	err := reg.Train(data, targets)
	if err != nil {
		fmt.Println("oops")
	}

	xTest := make([][]float64, 0)
	xTest = append(xTest, data.GetRow(25))

	yTest := targets.GetItem(25)

	preds := reg.Predict(ndarray.NewMatrix(xTest))

	fmt.Printf("true: %v preds: %v", yTest, preds)
}

func exampleFunc(x1, x2 float64) float64 {
	w1 := 100.
	w2 := 20.
	b := 1.

	return w1*x1 + w2*x2 + b
}
