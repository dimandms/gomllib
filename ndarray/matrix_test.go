package ndarray

import (
	"reflect"
	"testing"
)

func TestMatrix_Shape(t *testing.T) {
	tests := []struct {
		name  string
		m     *Matrix
		want1 int
		want2 int
	}{
		{
			"happy path 2x2",
			NewMatrix([][]float64{
				{1.0, 2.0},
				{3.0, 4.0},
			}),
			2,
			2,
		},
		{
			"happy path 3x2",
			NewMatrix([][]float64{
				{1.0, 2.0},
				{3.0, 4.0},
				{3.0, 4.0},
			}),
			3,
			2,
		},
		{
			"empty",
			NewMatrix([][]float64{}),
			0,
			0,
		},
		{
			"1D (single column)",
			NewMatrix([][]float64{
				{1.0},
				{3.0},
				{3.0},
			}),
			3,
			1,
		},
		{
			"1D (single row)",
			NewMatrix([][]float64{
				{1.0, 2.0, 3.0},
			}),
			1,
			3,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, got1 := tt.m.Shape()
			if got != tt.want1 {
				t.Errorf("Matrix.Shape() got = %v, want %v", got, tt.want1)
			}
			if got1 != tt.want2 {
				t.Errorf("Matrix.Shape() got1 = %v, want %v", got1, tt.want2)
			}
		})
	}
}

func TestMatrix_GetRow(t *testing.T) {
	type args struct {
		index int
	}
	tests := []struct {
		name    string
		m       *Matrix
		args    args
		want    []float64
		wantErr bool
	}{
		{
			"happy path",
			NewMatrix([][]float64{
				{1.0, 2.0},
				{3.0, 4.0},
				{3.0, 4.0},
			}),
			args{0},
			[]float64{1.0, 2.0},
			false,
		},
		{
			"wrong index",
			NewMatrix([][]float64{
				{1.0, 2.0},
				{3.0, 4.0},
				{3.0, 4.0},
			}),
			args{3},
			nil,
			true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.m.GetRow(tt.args.index)
			if (err != nil) != tt.wantErr {
				t.Errorf("Matrix.GetRow() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Matrix.GetRow() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMatrix_DotVector(t *testing.T) {
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
			NewMatrix([][]float64{
				{1.0, 2.0},
				{3.0, 4.0},
				{3.0, 4.0},
			}),
			args{NewVector([]float64{1.0, 1.0})},
			NewVector([]float64{3.0, 7.0, 7.0}),
			false,
		},
		{
			"incompatible shape",
			NewMatrix([][]float64{
				{1.0, 2.0},
				{3.0, 4.0},
				{3.0, 4.0},
			}),
			args{NewVector([]float64{1.0, 1.0, 1.0})},
			nil,
			true,
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
				{3.0, 4.0},
			}),
			NewMatrix([][]float64{
				{1.0, 3.0, 3.0},
				{2.0, 4.0, 4.0},
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

func TestMatrix_ExtendWith(t *testing.T) {
	type args struct {
		v *Vector
	}
	tests := []struct {
		name    string
		m       *Matrix
		args    args
		want    *Matrix
		wantErr bool
	}{
		{
			"happy path",
			NewMatrix([][]float64{
				{1.0, 3.0},
				{2.0, 4.0},
			}),
			args{NewVector([]float64{1.0, 2.0})},
			NewMatrix([][]float64{
				{1.0, 3.0, 1.0},
				{2.0, 4.0, 2.0},
			}),
			false,
		},
		{
			"wrong shapes",
			NewMatrix([][]float64{
				{1.0, 3.0},
				{2.0, 4.0},
			}),
			args{NewVector([]float64{1.0, 2.0, 3.0})},
			nil,
			true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.m.ExtendWith(tt.args.v)
			if (err != nil) != tt.wantErr {
				t.Errorf("Matrix.ExtendWith() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Matrix.ExtendWith() = %v, want %v", got, tt.want)
			}
		})
	}
}
