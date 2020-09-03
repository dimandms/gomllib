package ndarray

import (
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
