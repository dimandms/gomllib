package main

import (
	"fmt"
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
		x1 := float64(float64(n) / float64(numOfPoints))
		X = append(X, []float64{x1})
		y = append(y, exampleFunc(x1)+rand.Float64()/20)
		n += 1
	}

	reg := regression.NewLinearRegressor(true)

	scaler := ndarray.NewStandardScaler()
	data := scaler.FitTransform(ndarray.NewMatrix(X))
	targers := ndarray.NewVector(y).ScaleStandard()

	err := reg.Train(data, targers)
	if err != nil {
		fmt.Println("oops")
	}

	preds := reg.Predict(ndarray.NewMatrix([][]float64{data.GetRow(10)}))

	fmt.Printf("true: %v preds: %v", targers.GetItem(10), preds)
}

func exampleFunc(x1 float64) float64 {
	w1 := 100.
	b := 10.

	return w1*x1 + b
}
