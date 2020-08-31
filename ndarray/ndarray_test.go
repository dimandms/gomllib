package ndarray

import (
	"reflect"
	"testing"
)

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
			"happy path 2x2",
			NewMatrix([][]float64{
				{1.0, 2.0},
				{3.0, 4.0},
			}),
			NewMatrix([][]float64{
				{1.0, 3.0},
				{2.0, 4.0},
			}),
		},
		{
			"happy path 3x2",
			NewMatrix([][]float64{
				{1.0, 2.0},
				{3.0, 4.0},
				{5.0, 6.0},
			}),
			NewMatrix([][]float64{
				{1.0, 3.0, 5.0},
				{2.0, 4.0, 6.0},
			}),
		},
		{
			"happy path 1x3",
			NewMatrix([][]float64{
				{1.0, 2.0, 3.0},
			}),
			NewMatrix([][]float64{
				{1.0},
				{2.0},
				{3.0},
			}),
		},
		{
			"happy path 3x1",
			NewMatrix([][]float64{
				{1.0},
				{2.0},
				{3.0},
			}),
			NewMatrix([][]float64{
				{1.0, 2.0, 3.0},
			}),
		},
		{
			"empty item",
			NewMatrix([][]float64{
				{},
				{2.0},
				{3.0},
			}),
			NewMatrix([][]float64{
				{0.0, 2.0, 3.0},
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

func TestMatrix_Shape(t *testing.T) {
	tests := []struct {
		name       string
		m          *Matrix
		wantFirst  int
		wantSecond int
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
			"emtpy",
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
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, got1 := tt.m.Shape()
			if got != tt.wantFirst {
				t.Errorf("Matrix.Shape() got = %v, want %v", got, tt.wantFirst)
			}
			if got1 != tt.wantSecond {
				t.Errorf("Matrix.Shape() got1 = %v, want %v", got1, tt.wantSecond)
			}
		})
	}
}

func TestMatrix_getData(t *testing.T) {
	tests := []struct {
		name string
		m    *Matrix
		want [][]float64
	}{
		{
			name: "nil data slice",
			m: &Matrix{
				data: nil,
			},
			want: [][]float64{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.m.getData(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Matrix.getData() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMatrix_ExtendWith(t *testing.T) {
	type args struct {
		v *Vector
	}
	tests := []struct {
		name string
		m    *Matrix
		args args
		want *Matrix
	}{
		{"happy path",
			NewMatrix([][]float64{
				{1.0, 2.0},
				{3.0, 4.0},
			}),
			args{NewVector([]float64{
				4.0, 4.0,
			})},
			NewMatrix([][]float64{
				{1.0, 2.0, 4.0},
				{3.0, 4.0, 4.0},
			})},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.m.ExtendWith(tt.args.v); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Matrix.ExtendWith() = %v, want %v", got, tt.want)
			}
		})
	}
}
