package ndarray

import (
	"math"
	"reflect"
	"testing"
)

const float64EqualityThreshold = 1e-9

func equal(a, b float64) bool {
	return math.Abs(a-b) <= float64EqualityThreshold
}

func TestMatrix_Dot(t *testing.T) {
	type args struct {
		v *Vector
	}
	tests := []struct {
		name    string
		m       *Matrix
		args    args
		want    *Vector
		wantErr bool
	}{
		{
			"happy path",
			&Matrix{
				data: [][]float64{
					{1.0, 2.0, 3.0},
					{-1.0, 2.0, 3.0},
					{-1.0, -2.0, 3.0},
					{-1.0, 2.0, -3.0},
				},
			},
			args{&Vector{data: []float64{1.0, 2.0, 3.0}}},
			&Vector{data: []float64{14.0, 12.0, 4.0, -6.0}},
			false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.m.DotVector(tt.args.v)
			if (err != nil) != tt.wantErr {
				t.Errorf("Matrix.DotVector() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Matrix.DotVector() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_scalarMultiplication(t *testing.T) {
	type args struct {
		a *Vector
		b *Vector
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{
			"happy path test",
			args{a: NewVector([]float64{1.0, 2.0, 3.0}), b: NewVector([]float64{3.0, 2.0, 1.0})},
			10.0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := scalarMultiplication(tt.args.a, tt.args.b)
			if !equal(got, tt.want) {
				t.Errorf("scalarMultiplication() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMatrix_Transpose(t *testing.T) {
	tests := []struct {
		name string
		m    *Matrix
		want *Matrix
	}{
		{
			"happy path",
			NewMatrix([][]float64{
				{1.0, 2.0},
				{3.0, 4.0},
			}),
			NewMatrix([][]float64{
				{1.0, 3.0},
				{2.0, 4.0},
			}),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.m.Transpose(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Matrix.Transpose() = %v, want %v", got, tt.want)
			}
		})
	}
}
