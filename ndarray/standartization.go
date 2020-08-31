package ndarray

import "math"

type Transformer interface {
	Fit(objects *Matrix) *Matrix
	Transform(objects *Matrix) *Matrix
}

type StandardScaler struct {
	mean               []float64
	standartdDeviation []float64
}

func NewStandardScaler() *StandardScaler {
	return &StandardScaler{}
}

func (s *StandardScaler) Fit(objects *Matrix) {
	data := objects.Transpose().getData()

	for _, feature := range data {
		featureMean := mean(feature)
		featureStandartdDeviation := std(feature, featureMean)
		s.mean = append(s.mean, featureMean)
		s.standartdDeviation = append(s.mean, featureStandartdDeviation)
	}
}

func (s *StandardScaler) Transform(objects *Matrix) *Matrix {
	result := make([][]float64, 0)

	for i, object := range objects.getData() {
		transformedObject := make([]float64, 0)
		featureMean := s.mean[i]
		featureStandartdDeviation := s.standartdDeviation[i]

		for _, featureValue := range object {
			transformedValue := (featureValue - featureMean) / featureStandartdDeviation
			transformedObject = append(transformedObject, transformedValue)
		}

		result = append(result, transformedObject)

	}

	return NewMatrix(result)
}

func mean(data []float64) float64 {
	result := 0.0
	for _, item := range data {
		result += item
	}

	return result / float64(len(data))
}

func std(data []float64, mean float64) float64 {
	result := 0.0
	for _, item := range data {
		result += math.Pow(item-mean, 2)
	}

	return math.Sqrt(result / float64(len(data)))
}
