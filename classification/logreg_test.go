package logreg

import (
	"math"
	"testing"
)

const float64EqualityThreshold = 1e-9

func equal(a, b float64) bool {
	return math.Abs(a-b) <= float64EqualityThreshold
}

func Test_scalarMultiplication(t *testing.T) {
	type args struct {
		a []float64
		b []float64
	}
	tests := []struct {
		name    string
		args    args
		want    float64
		wantErr bool
	}{
		{
			"happy path test",
			args{a: []float64{1.0, 2.0, 3.0}, b: []float64{3.0, 2.0, 1.0}},
			10.0,
			false,
		},
		{
			"not equal lenghts",
			args{a: []float64{2.0, 3.0}, b: []float64{3.0, 2.0, 1.0}},
			0.0,
			true,
		},
		{
			"not equal lenghts 2",
			args{a: []float64{1.0, 2.0, 3.0}, b: []float64{2.0, 1.0}},
			0.0,
			true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := scalarMultiplication(tt.args.a, tt.args.b)
			if (err != nil) != tt.wantErr {
				t.Errorf("scalarMultiplication() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !equal(got, tt.want) {
				t.Errorf("scalarMultiplication() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestSquareError(t *testing.T) {
	type args struct {
		calculatedAnswer float64
		trueAnswer       float64
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{
			"happy path with return 0",
			args{5, 5},
			0.0,
		},
		{
			"happy path with return 1",
			args{4, 5},
			1.0,
		},
		{
			"happy path with return 36",
			args{1, 7},
			36.0,
		},
		{
			"happy path with single below zero first argument",
			args{-5, 1},
			36.0,
		},
		{
			"happy path with single below zero second argument",
			args{1, -5},
			36.0,
		},
		{
			"happy path with single below zero both argument",
			args{-1, -3},
			4.0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := squareError(tt.args.calculatedAnswer, tt.args.trueAnswer); got != tt.want {
				t.Errorf("SquareError() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMeanSquareError(t *testing.T) {
	type args struct {
		weights     []float64
		objects     [][]float64
		trueAnswers []float64
	}
	tests := []struct {
		name    string
		args    args
		want    float64
		wantErr bool
	}{
		{
			"happy path",
			args{
				[]float64{1.0, -1.0, 3.0},
				[][]float64{
					{1.0, 2.0, 3.0},
					{-1.0, 2.0, 3.0},
					{-1.0, -2.0, 3.0},
				},
				[]float64{8.0, 6.0, 10.0},
			},
			0.0,
			false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := MeanSquareError(tt.args.weights, tt.args.objects, tt.args.trueAnswers)
			if (err != nil) != tt.wantErr {
				t.Errorf("MeanSquareError() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("MeanSquareError() = %v, want %v", got, tt.want)
			}
		})
	}
}
