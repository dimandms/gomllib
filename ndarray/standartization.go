package ndarray

import "math"

const float64EqualityThreshold = 1e-5

// Transformer interface for transforming features
type Transformer interface {
	Fit(objects *Matrix) *Matrix
	Transform(objects *Matrix) *Matrix
}

// StandardScaler is type for scale features to standartization
type StandardScaler struct {
	mean               []float64
	standartdDeviation []float64
}

// NewStandardScaler is basic constructor fot StandardScaler
func NewStandardScaler() *StandardScaler {
	return &StandardScaler{}
}

// Fit is method to calculate mean and standard deviation
func (s *StandardScaler) Fit(objects *Matrix) {
	data := objects.Transpose().getData()

	for _, feature := range data {
		featureMean := mean(feature)
		featureStandartdDeviation := std(feature, featureMean)
		s.mean = append(s.mean, featureMean)
		s.standartdDeviation = append(s.standartdDeviation, featureStandartdDeviation)
	}
}

// Transform is method to transform data with calculated mean and standard deviation
func (s *StandardScaler) Transform(objects *Matrix) *Matrix {
	result := make([][]float64, 0)

	for _, object := range objects.getData() {
		transformedObject := make([]float64, 0)

		for i, featureValue := range object {
			featureMean := s.mean[i]
			featureStandartdDeviation := s.standartdDeviation[i]

			if !equal(featureStandartdDeviation, 0.0) {
				transformedValue := (featureValue - featureMean) / featureStandartdDeviation
				transformedObject = append(transformedObject, transformedValue)
			} else {
				transformedValue := (featureValue - featureMean) / 1.0
				transformedObject = append(transformedObject, transformedValue)
			}
		}

		result = append(result, transformedObject)

	}

	return NewMatrix(result)
}

// FitTransform combined Fit and Transform for input data
func (s *StandardScaler) FitTransform(objects *Matrix) *Matrix {
	s.Fit(objects)
	return s.Transform(objects)
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

func equal(a, b float64) bool {
	return math.Abs(a-b) <= float64EqualityThreshold
}
