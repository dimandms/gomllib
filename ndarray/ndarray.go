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

func NewVectorFrom(value float64, size int) (*Vector, error) {
	if size < 1 {
		return nil, fmt.Errorf("Allocation of the new *Vector failed. Size passed: %v", size)
	}

	data := make([]float64, size)
	for i := range data {
		data[i] = value
	}

	return &Vector{data}, nil
}

func (m *Matrix) Shape() (int, int) {
	rows := len(m.data)
	if rows > 0 {
		columns := len(m.data[0])
		return rows, columns
	}

	return 0, 0
}

func (v *Vector) Shape() int {
	return len(v.data)
}

func (m *Matrix) getData() ([][]float64, error) {
	if m.data != nil {
		return m.data, nil
	}

	return nil, fmt.Errorf("Get data from *Matrix failed: data field is a nil pointer")
}

func (v *Vector) getData() ([]float64, error) {
	if v.data != nil {
		return v.data, nil
	}

	return nil, fmt.Errorf("Get data from *Vector failed: data field is a nil pointer")
}

func (m *Matrix) GetRow(index int) ([]float64, error) {
	data, err := m.getData()
	if err != nil {
		return nil, fmt.Errorf("Get Row from *Matrix failed: %v", err)
	}

	dataLength := len(data)
	if index >= dataLength {
		return nil, fmt.Errorf("Get Row index=[%v] from *Matrix failed, number of rows: %v", index, dataLength)
	}

	return data[index], nil
}

func (v *Vector) GetItem(index int) (float64, error) {
	data, err := v.getData()
	if err != nil {
		return 0.0, fmt.Errorf("Get item from *Vector failed: %v", err)
	}

	dataLength := len(data)
	if index >= dataLength {
		return 0.0, fmt.Errorf("Get item index=[%v] from *Vector failed, lenth of data is: %v", index, dataLength)
	}

	return data[index], nil
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

func (v *Vector) MultiplicateBy(n float64) *Vector {
	result := make([]float64, 0)
	for _, value := range v.getData() {
		result = append(result, value*n)
	}

	return NewVector(result)
}

func (v *Vector) Copy() *Vector {
	data := v.getData()
	result := make([]float64, len(data))
	copy(result, data)

	return NewVector(result)
}

func (v *Vector) AbsMax() float64 {
	result := 0.0

	for _, value := range v.getData() {
		absValue := math.Abs(value)
		if absValue > result {
			result = absValue
		}
	}

	return result
}

func (m *Matrix) ExtendWith(v *Vector) *Matrix {
	result := make([][]float64, 0)

	column := v.getData()

	for i, rowItems := range m.getData() {
		result = append(result, append(rowItems, column[i]))
	}

	return NewMatrix(result)
}

func (v *Vector) ScaleStandard() *Vector {
	result := make([]float64, 0)

	vectorMean := mean(v.getData())
	vectorStandartdDeviation := std(v.getData(), vectorMean)

	for _, item := range v.getData() {
		if !equal(vectorStandartdDeviation, 0.0) {
			transformedValue := (item - vectorMean) / vectorStandartdDeviation
			result = append(result, transformedValue)
		} else {
			transformedValue := (item - vectorMean) / 1.0
			result = append(result, transformedValue)
		}

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
