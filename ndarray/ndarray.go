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

func NewMatrix(data [][]float64) *Matrix {
	return &Matrix{
		data: data,
	}
}

func NewVector(data []float64) *Vector {
	return &Vector{
		data: data,
	}
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

	//TODO: move checjing to scalarMultiplication func
	if columns != vectorSize {
		return nil, fmt.Errorf("Maxtix dot product with vector failed: incompatable shapes: %v, %v", columns, vectorSize)
	}

	result := make([]float64, rows)
	for i, rowItems := range m.getData() {
		rowVector := NewVector(rowItems)
		result[i] = scalarMultiplication(rowVector, v)
	}

	return NewVector(result), nil
}

func scalarMultiplication(a, b *Vector) float64 {
	var result float64

	aData := a.getData()
	bData := b.getData()

	for i, item := range aData {
		result += item * bData[i]
	}

	return result
}
