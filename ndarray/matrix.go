package ndarray

import (
	"fmt"
)

type Matrix struct {
	data [][]float64
}

func NewMatrix(data [][]float64) *Matrix {
	return &Matrix{
		data: data,
	}
}

func (m *Matrix) Shape() (int, int) {
	data := m.getData()
	rowsNumber := len(data)
	if rowsNumber > 0 {
		columnsNumber := len(data[0])
		return rowsNumber, columnsNumber
	}

	return 0, 0
}

func (m *Matrix) getData() [][]float64 {
	//TODO: getter for future concurrency access to data
	return m.data
}

func (m *Matrix) GetRow(index int) ([]float64, error) {
	data := m.getData()

	dataLength := len(data)
	if index >= dataLength {
		return nil, fmt.Errorf("Get Row index=[%v] from *Matrix failed, number of rows: %v", index, dataLength)
	}

	return data[index], nil
}

// TODO: switch to argument as an interface Tensor? with Shape(), GetRow()
// Matrix and Vector types has 2 method to work inside Dot()
func (m *Matrix) DotVector(v *Vector) (*Vector, error) {
	result := make([]float64, 0)

	for _, row := range m.getData() {
		mult, err := scalarMultiplication(NewVector(row), v)
		if err != nil {
			return nil, fmt.Errorf("DotVector failed: %v", err)
		}
		result = append(result, mult)
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

func (m *Matrix) ExtendWith(v *Vector) (*Matrix, error) {
	rowsNumber, _ := m.Shape()
	itemsNumber := v.Shape()

	if rowsNumber != itemsNumber {
		return nil, fmt.Errorf("Extend matrix with new column failed: rows number != itemsNUmber: %v != %v", rowsNumber, itemsNumber)
	}

	result := make([][]float64, rowsNumber)
	newColumn := v.getData()
	for i, row := range m.getData() {
		result[i] = append(row, newColumn[i])
	}

	return NewMatrix(result), nil
}
