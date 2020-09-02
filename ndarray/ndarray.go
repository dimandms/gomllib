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

func (m *Matrix) Shape() (int, int, error) {
	data, err := m.getData()
	if err != nil {
		return 0, 0, err
	}

	rowsNumber := len(data)
	if rowsNumber > 0 {
		columnsNumber := len(data[0])
		return rowsNumber, columnsNumber, nil
	}

	return 0, 0, nil
}

func (v *Vector) Shape() (int, error) {
	data, err := v.getData()
	if err != nil {
		return 0, err
	}

	return len(data), nil
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

//TODO: switch to argument as a interface Tensor? with Shape(), GetRow()
// Matrix and Vector types has 2 method to work inside Dot()
func (m *Matrix) DotVector(v *Vector) (*Vector, error) {
	result := make([]float64, 0)

	rows, err := m.getData()
	if err != nil {
		return nil, fmt.Errorf("DotVector call failed: %v", err)
	}

	for _, row := range rows {
		mult, err := scalarMultiplication(NewVector(row), v)
		if err != nil {
			return nil, fmt.Errorf("DotVector failed: %v", err)
		}
		result = append(result, mult)
	}

	return NewVector(result), nil
}

func (m *Matrix) Transpose() (*Matrix, error) {
	mData, err := m.getData()
	if err != nil {
		return nil, fmt.Errorf("DotVector failed: %v", err)
	}

	k, z, err := m.Shape()
	if err != nil {
		return nil, fmt.Errorf("DotVector failed: %v", err)
	}

	result := make([][]float64, z)
	for i := range result {
		result[i] = make([]float64, k)
	}

	for i, row := range mData {
		for j := range row {
			result[j][i] = mData[i][j]
		}
	}

	return NewMatrix(result), nil
}

func (v *Vector) DotVector(u *Vector) (float64, error) {
	mult, err := scalarMultiplication(v, u)
	if err != nil {
		return 0.0, fmt.Errorf("DotVector failed: %v", err)
	}

	return mult, err
}

func (v *Vector) SubVector(u *Vector) (*Vector, error) {
	vData, err := v.getData()
	if err != nil {
		return nil, fmt.Errorf("SubVector failed cause v Vector: %v", err)
	}

	uData, err := u.getData()
	if err != nil {
		return nil, fmt.Errorf("SubVector failed cause u Vector: %v", err)
	}

	vLength := len(vData)
	uLength := len(uData)

	if vLength != uLength {
		return nil, fmt.Errorf("vector sub with vector failed: incompatable shapes: %v, %v", vLength, uLength)
	}

	result := make([]float64, vLength)
	for i := range vData {
		result[i] = vData[i] - uData[i]
	}

	return NewVector(result), nil
}

func (v *Vector) AddVector(u *Vector) (*Vector, error) {
	vData, err := v.getData()
	if err != nil {
		return nil, fmt.Errorf("vector add failed cause v Vector: %v", err)
	}

	uData, err := u.getData()
	if err != nil {
		return nil, fmt.Errorf("vector add failed cause u Vector: %v", err)
	}

	vLength := len(vData)
	uLength := len(uData)

	if vLength != uLength {
		return nil, fmt.Errorf("vector add with vector failed: incompatable shapes: %v, %v", vLength, uLength)
	}

	result := make([]float64, vLength)
	for i := range vData {
		result[i] = vData[i] + uData[i]
	}

	return NewVector(result), nil
}

func (v *Vector) Sum() (float64, error) {
	var result float64

	data, err := v.getData()
	if err != nil {
		return 0.0, fmt.Errorf("vector sum failed: %v", err)
	}

	for _, value := range data {
		result += value
	}

	return result, nil
}

func (v *Vector) Pow(n float64) (*Vector, error) {
	data, err := v.getData()
	if err != nil {
		return nil, fmt.Errorf("vector sum failed: %v", err)
	}

	result := make([]float64, len(data))
	for i, value := range data {
		result[i] = math.Pow(value, n)
	}

	return NewVector(result), nil
}

func (v *Vector) MultiplicateBy(n float64) (*Vector, error) {
	data, err := v.getData()
	if err != nil {
		return nil, fmt.Errorf("vector mult by failed: %v", err)
	}

	result := make([]float64, len(data))

	for i, value := range data {
		result[i] = value * n
	}

	return NewVector(result), nil
}

func (v *Vector) Copy() (*Vector, error) {
	data, err := v.getData()
	if err != nil {
		return nil, fmt.Errorf("vector copy by failed: %v", err)
	}

	result := make([]float64, len(data))
	copy(result, data)

	return NewVector(result), nil
}

func (v *Vector) AbsMax() (float64, error) {
	var result float64

	data, err := v.getData()
	if err != nil {
		return 0.0, fmt.Errorf("vector abs max failed: %v", err)
	}

	for _, value := range data {
		absValue := math.Abs(value)
		if absValue > result {
			result = absValue
		}
	}

	return result, nil
}

func (m *Matrix) ExtendWith(v *Vector) (*Matrix, error) {
	rowsNumber, _, err := m.Shape()
	if err != nil {
		return nil, fmt.Errorf("Extend matrix with new column failed: %v", err)
	}

	itemsNumber, err := v.Shape()
	if err != nil {
		return nil, fmt.Errorf("Extend matrix with new column failed: %v", err)
	}

	if rowsNumber != itemsNumber {
		return nil, fmt.Errorf("Extend matrix with new column failed: rows number != itemsNUmber: %v != %v", rowsNumber, itemsNumber)
	}

	newColumn, err := v.getData()
	if err != nil {
		return nil, fmt.Errorf("Extend matrix with new column failed: %v", err)
	}

	rows, err := m.getData()
	if err != nil {
		return nil, fmt.Errorf("Extend matrix with new column failed: %v", err)
	}

	result := make([][]float64, rowsNumber)
	for i, row := range rows {
		result[i] = append(row, newColumn[i])
	}

	return NewMatrix(result), nil
}

func (v *Vector) ScaleStandard() (*Vector, float64, float64, error) {
	data, err := v.getData()
	if err != nil {
		return nil, 0.0, 0.0, fmt.Errorf("Extend matrix with new column failed: %v", err)
	}

	vectorMean := mean(data)
	vectorStandartdDeviation := std(data, vectorMean)

	result := make([]float64, len(data))
	if !equal(vectorStandartdDeviation, 0.0) {
		for i, item := range data {
			result[i] = (item - vectorMean) / vectorStandartdDeviation
		}
	} else {
		for i, item := range data {
			result[i] = item - vectorMean
		}
	}

	return NewVector(result), vectorMean, vectorStandartdDeviation, nil
}

//TODO: works with data length not Shape(), because current
//Vector implementation do not have rows, columns and Transpose()
func scalarMultiplication(a, b *Vector) (float64, error) {
	var result float64

	aData, err := a.getData()
	if err != nil {
		return 0.0, fmt.Errorf("Scalar multiplication failed: getData from a vector failed")
	}

	bData, err := b.getData()
	if err != nil {
		return 0.0, fmt.Errorf("Scalar multiplication failed: getData from b vector failed")
	}

	aLength := len(a.data)
	bLength := len(b.data)
	if aLength != bLength {
		return 0.0, fmt.Errorf("Scalar multiplication failed: incompatable data lengths: %v, %v", aLength, bLength)
	}

	for i, item := range aData {
		result += item * bData[i]
	}

	return result, nil
}
