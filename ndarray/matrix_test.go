package ndarray

import "testing"

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
