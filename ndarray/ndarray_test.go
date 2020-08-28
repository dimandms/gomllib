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
