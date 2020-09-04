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
		x2 := float64(float64(n) / float64(numOfPoints))
		X = append(X, []float64{x1, x2})
		y = append(y, exampleFunc(x1, x2)+rand.Float64()/20)
		n += 1
	}

	reg := regression.NewLinearRegressor(true)

	scaler := ndarray.NewStandardScaler()
	data := scaler.FitTransform(ndarray.NewMatrix(X))
	targers, _, _ := ndarray.NewVector(y).ScaleStandard()

	err := reg.Train(data, targers)
	if err != nil {
		fmt.Println("oops")
	}

	row, _ := data.GetRow(15)
	preds := reg.Predict(ndarray.NewMatrix([][]float64{row}))
	trueValue, _ := targers.GetItem(15)

	fmt.Printf("true: %v preds: %v \n", trueValue, preds)
	fmt.Printf("weights: %v ", reg.Weights)
}

func exampleFunc(x1, x2 float64) float64 {
	w1 := 100.
	w2 := 200.
	b := 10.

	return w1*x1 + w2*x2 + b
}
