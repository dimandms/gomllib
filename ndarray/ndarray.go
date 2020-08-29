package ndarray

import (
	"fmt"
	"math"
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

func NewVectorFrom(value float64, size int) *Vector {
	data := make([]float64, size)

	for i := range data {
		data[i] = value
	}

	return &Vector{data}
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

func (m *Matrix) Transpose() *Matrix {
	k, z := m.Shape()

	result := make([][]float64, z)
	for i := range result {
		result[i] = make([]float64, k)
	}

	mData := m.getData()

	for i, row := range mData {
		for j := range row {
			result[j][i] = mData[i][j]
		}
	}

	return NewMatrix(result)
}

func (v *Vector) DotVector(u *Vector) (float64, error) {
	if v.Shape() != u.Shape() {
		return 0.0, fmt.Errorf("vector dot product with vector failed: incompatable shapes: %v, %v", v.Shape(), u.Shape())
	}

	return scalarMultiplication(v, u), nil
}

func (v *Vector) SubVector(u *Vector) (*Vector, error) {
	if v.Shape() != u.Shape() {
		return nil, fmt.Errorf("vector sub with vector failed: incompatable shapes: %v, %v", v.Shape(), u.Shape())
	}

	result := make([]float64, v.Shape())
	vData := v.getData()
	uData := u.getData()

	for i := range vData {
		result[i] = vData[i] - uData[i]
	}

	return NewVector(result), nil
}

func (v *Vector) AddVector(u *Vector) (*Vector, error) {
	if v.Shape() != u.Shape() {
		return nil, fmt.Errorf("vector sub with vector failed: incompatable shapes: %v, %v", v.Shape(), u.Shape())
	}

	result := make([]float64, v.Shape())
	vData := v.getData()
	uData := u.getData()

	for i := range vData {
		result[i] = vData[i] + uData[i]
	}

	return NewVector(result), nil
}

func (v *Vector) Sum() float64 {
	result := 0.0
	for _, value := range v.getData() {
		result += value
	}

	return result
}

func (v *Vector) Pow(n float64) *Vector {
	result := make([]float64, 0)
	for _, value := range v.getData() {
		result = append(result, math.Pow(value, n))
	}

	return NewVector(result)
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
