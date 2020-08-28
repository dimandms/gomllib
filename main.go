package main

import (
	"fmt"

	clf "github.com/dimandms/gomllib/classification"
)

func main() {
	clf := clf.New(false)

	x := [][]float64{
		{1.0, 2.0, 3.0},
		{-1.0, 2.0, 3.0},
		{-1.0, -2.0, 3.0},
	}
	y := []float64{1.0, 0.0, 0.0}

	err := clf.Train(x, y)
	if err != nil {
		fmt.Println("error")
	}
	fmt.Println(clf.Weights)
}
