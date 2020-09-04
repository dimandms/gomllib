package ndarray

import (
	"reflect"
	"testing"
)

func TestVector_Shape(t *testing.T) {
	tests := []struct {
		name string
		v    *Vector
		want int
	}{
		{
			"happy path",
			NewVector([]float64{1.0, 2.0, 3.0}),
			3,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.v.Shape(); got != tt.want {
				t.Errorf("Vector.Shape() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestVector_GetItem(t *testing.T) {
	type args struct {
		index int
	}
	tests := []struct {
		name    string
		v       *Vector
		args    args
		want    float64
		wantErr bool
	}{
		{
			"happy path",
			NewVector([]float64{1.0, 2.0, 3.0}),
			args{1},
			2.0,
			false,
		},
		{
			"wrong index",
			NewVector([]float64{1.0, 2.0, 3.0}),
			args{3},
			0.0,
			true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.v.GetItem(tt.args.index)
			if (err != nil) != tt.wantErr {
				t.Errorf("Vector.GetItem() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("Vector.GetItem() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestVector_DotVector(t *testing.T) {
	type args struct {
		u *Vector
	}
	tests := []struct {
		name    string
		v       *Vector
		args    args
		want    float64
		wantErr bool
	}{
		{
			"happy path",
			NewVector([]float64{1.0, 2.0, 3.0}),
			args{NewVector([]float64{1.0, 2.0, 3.0})},
			14.0,
			false,
		},
		{
			"wrong shapes",
			NewVector([]float64{1.0, 2.0}),
			args{NewVector([]float64{1.0, 2.0, 3.0})},
			0.0,
			true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.v.DotVector(tt.args.u)
			if (err != nil) != tt.wantErr {
				t.Errorf("Vector.DotVector() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("Vector.DotVector() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestVector_SubVector(t *testing.T) {
	type args struct {
		u *Vector
	}
	tests := []struct {
		name    string
		v       *Vector
		args    args
		want    *Vector
		wantErr bool
	}{
		{
			"happy path",
			NewVector([]float64{1.0, 2.0, 3.0}),
			args{NewVector([]float64{1.0, 2.0, 3.0})},
			NewVector([]float64{0.0, 0.0, 0.0}),
			false,
		},
		{
			"wrong shapes",
			NewVector([]float64{1.0, 2.0}),
			args{NewVector([]float64{1.0, 2.0, 3.0})},
			nil,
			true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.v.SubVector(tt.args.u)
			if (err != nil) != tt.wantErr {
				t.Errorf("Vector.SubVector() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Vector.SubVector() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestVector_AddVector(t *testing.T) {
	type args struct {
		u *Vector
	}
	tests := []struct {
		name    string
		v       *Vector
		args    args
		want    *Vector
		wantErr bool
	}{
		{
			"happy path",
			NewVector([]float64{1.0, 2.0, 3.0}),
			args{NewVector([]float64{1.0, 2.0, 3.0})},
			NewVector([]float64{2.0, 4.0, 6.0}),
			false,
		},
		{
			"wrong shapes",
			NewVector([]float64{1.0, 2.0}),
			args{NewVector([]float64{1.0, 2.0, 3.0})},
			nil,
			true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.v.AddVector(tt.args.u)
			if (err != nil) != tt.wantErr {
				t.Errorf("Vector.AddVector() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Vector.AddVector() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestVector_Sum(t *testing.T) {
	tests := []struct {
		name string
		v    *Vector
		want float64
	}{
		{
			"happy path",
			NewVector([]float64{1.0, 2.0, 3.0}),
			6.0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.v.Sum(); got != tt.want {
				t.Errorf("Vector.Sum() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestVector_Pow(t *testing.T) {
	type args struct {
		n float64
	}
	tests := []struct {
		name string
		v    *Vector
		args args
		want *Vector
	}{
		{
			"happy path",
			NewVector([]float64{1.0, 2.0, 3.0}),
			args{2.0},
			NewVector([]float64{1.0, 4.0, 9.0}),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.v.Pow(tt.args.n); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Vector.Pow() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestVector_MultiplicateBy(t *testing.T) {
	type args struct {
		n float64
	}
	tests := []struct {
		name string
		v    *Vector
		args args
		want *Vector
	}{
		{
			"happy path",
			NewVector([]float64{1.0, 2.0, 3.0}),
			args{2.0},
			NewVector([]float64{2.0, 4.0, 6.0}),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.v.MultiplicateBy(tt.args.n); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Vector.MultiplicateBy() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestVector_AbsMax(t *testing.T) {
	tests := []struct {
		name string
		v    *Vector
		want float64
	}{
		{
			"happy path",
			NewVector([]float64{-4.0, 2.0, 3.0}),
			4.0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.v.AbsMax(); got != tt.want {
				t.Errorf("Vector.AbsMax() = %v, want %v", got, tt.want)
			}
		})
	}
}
