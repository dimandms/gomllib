package ndarray

import (
	"testing"
)

func Test_mean(t *testing.T) {
	type args struct {
		data []float64
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{
			"happy path",
			args{[]float64{1.0, 2.0, 3.0}},
			2.0,
		},
		{
			"happy path 1",
			args{[]float64{1.0, 1.0, 1.0}},
			1.0,
		},
		{
			"happy path 2",
			args{[]float64{-2.0, -1.0, 1.0, 2.0}},
			0.0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := mean(tt.args.data); !equal(got, tt.want) {
				t.Errorf("mean() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_std(t *testing.T) {
	type args struct {
		data []float64
		mean float64
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{
			"happy path",
			args{[]float64{1.0, 1.0, 1.0}, 1.0},
			0.0,
		},
		{
			"happy path",
			args{[]float64{-2.0, -1.0, 1.0, 2.0}, 0.0},
			1.58114,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := std(tt.args.data, tt.args.mean); !equal(got, tt.want) {
				t.Errorf("std() = %v, want %v", got, tt.want)
			}
		})
	}
}
