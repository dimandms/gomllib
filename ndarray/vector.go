package ndarray

import (
	"fmt"
	"math"
)

type Vector struct {
	data []float64
}

func NewVector(data []float64) *Vector {
	return &Vector{
		data: data,
	}
}

func NewVectorFrom(value float64, size uint) *Vector {
	data := make([]float64, size)
	for i := range data {
		data[i] = value
	}

	return &Vector{data}
}

func (v *Vector) Shape() int {
	data := v.getData()

	return len(data)
}

func (v *Vector) getData() []float64 {
	//TODO: getter for future concurrency access to data
	return v.data
}

func (v *Vector) GetItem(index int) (float64, error) {
	data := v.getData()

	dataLength := len(data)
	if index >= dataLength {
		return 0.0, fmt.Errorf("Get item index=[%v] from *Vector failed, lenth of data is: %v", index, dataLength)
	}

	return data[index], nil
}

func (v *Vector) DotVector(u *Vector) (float64, error) {
	mult, err := scalarMultiplication(v, u)
	if err != nil {
		return 0.0, fmt.Errorf("DotVector failed: %v", err)
	}

	return mult, err
}

func (v *Vector) SubVector(u *Vector) (*Vector, error) {
	vData := v.getData()
	uData := u.getData()

	vLength := len(vData)
	uLength := len(uData)

	if vLength != uLength {
		return nil, fmt.Errorf("vector sub with vector failed: incompatible shapes: %v, %v", vLength, uLength)
	}

	result := make([]float64, vLength)
	for i := range vData {
		result[i] = vData[i] - uData[i]
	}

	return NewVector(result), nil
}

func (v *Vector) AddVector(u *Vector) (*Vector, error) {
	vData := v.getData()
	uData := u.getData()

	vLength := len(vData)
	uLength := len(uData)

	if vLength != uLength {
		return nil, fmt.Errorf("vector add with vector failed: incompatible shapes: %v, %v", vLength, uLength)
	}

	result := make([]float64, vLength)
	for i := range vData {
		result[i] = vData[i] + uData[i]
	}

	return NewVector(result), nil
}

func (v *Vector) Sum() float64 {
	var result float64

	for _, value := range v.getData() {
		result += value
	}

	return result
}

func (v *Vector) Pow(n float64) *Vector {
	data := v.getData()
	result := make([]float64, len(data))

	for i, value := range data {
		result[i] = math.Pow(value, n)
	}

	return NewVector(result)
}

func (v *Vector) MultiplicateBy(n float64) *Vector {
	data := v.getData()
	result := make([]float64, len(data))

	for i, value := range data {
		result[i] = value * n
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
	var result float64
	data := v.getData()

	for _, value := range data {
		absValue := math.Abs(value)
		if absValue > result {
			result = absValue
		}
	}

	return result
}

func (v *Vector) ScaleStandard() (*Vector, float64, float64) {
	data := v.getData()

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

	return NewVector(result), vectorMean, vectorStandartdDeviation
}

//TODO: works with data length not Shape(), because current
//Vector implementation do not have rows, columns and Transpose()
func scalarMultiplication(a, b *Vector) (float64, error) {
	var result float64
	aData := a.getData()
	bData := b.getData()

	aLength := len(a.data)
	bLength := len(b.data)
	if aLength != bLength {
		return 0.0, fmt.Errorf("Scalar multiplication failed: incompatible data lengths: %v, %v", aLength, bLength)
	}

	for i, item := range aData {
		result += item * bData[i]
	}

	return result, nil
}
