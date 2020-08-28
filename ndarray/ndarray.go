package ndarray

import (
	"fmt"
)

type Matrix struct {
	data [][]float64
}

type Vector struct {
	data []float64
}

func (m *Matrix) Shape() (int, int) {
	return len(m.data), len(m.data[0])
}

func (v *Vector) Shape() int {
	return len(v.data)
}

func (m *Matrix) getData() [][]float64 {
	return m.data
}

func (v *Vector) getData() []float64 {
	return v.data
}

func (m *Matrix) DotVector(v *Vector) (*Vector, error) {
	rows, columns := m.Shape()
	vectorSize := v.Shape()

	if columns != vectorSize {
		return nil, fmt.Errorf("Maxtix dot product with vector failed: incompatable shapes: %v, %v", columns, vectorSize)
	}

	result := make([]float64, rows)
	for i, rowItems := range m.getData() {
		result[i] = scalarMultiplication(rowItems, v.data)

	}

	return &Vector{data: result}, nil
}

func scalarMultiplication(a, b []float64) float64 {
	var result float64

	for i, item := range a {
		result += item * b[i]
	}

	return result
}
