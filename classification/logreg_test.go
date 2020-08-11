package logreg

import (
	"math"
	"reflect"
	"testing"
)

const float64EqualityThreshold = 1e-9

func equal(a, b float64) bool {
	return math.Abs(a-b) <= float64EqualityThreshold
}

func TestNew(t *testing.T) {
	type args struct {
		fit_intercept bool
	}
	tests := []struct {
		name string
		args args
		want *LogisticRegression
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := New(tt.args.fit_intercept); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("New() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestLogisticRegression_Fit(t *testing.T) {
	type args struct {
		trainX float64
		trainY float64
	}
	tests := []struct {
		name    string
		lr      LogisticRegression
		args    args
		wantErr bool
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.lr.Fit(tt.args.trainX, tt.args.trainY); (err != nil) != tt.wantErr {
				t.Errorf("LogisticRegression.Fit() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestMeanSquareError(t *testing.T) {
	type args struct {
		weights     []float64
		objects     [][]float64
		trueAnswers []float64
	}
	tests := []struct {
		name    string
		args    args
		want    float64
		wantErr bool
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := MeanSquareError(tt.args.weights, tt.args.objects, tt.args.trueAnswers)
			if (err != nil) != tt.wantErr {
				t.Errorf("MeanSquareError() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("MeanSquareError() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestSquareError(t *testing.T) {
	type args struct {
		weights        []float64
		featuresValues []float64
		trueAnswer     float64
	}
	tests := []struct {
		name    string
		args    args
		want    float64
		wantErr bool
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := SquareError(tt.args.weights, tt.args.featuresValues, tt.args.trueAnswer)
			if (err != nil) != tt.wantErr {
				t.Errorf("SquareError() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("SquareError() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_scalarMultiplication(t *testing.T) {
	type args struct {
		a []float64
		b []float64
	}
	tests := []struct {
		name    string
		args    args
		want    float64
		wantErr bool
	}{
		{
			"happy path test",
			args{a: []float64{1.0, 2.0, 3.0}, b: []float64{3.0, 2.0, 1.0}},
			10.0,
			false,
		},
		{
			"not equal lenghts",
			args{a: []float64{2.0, 3.0}, b: []float64{3.0, 2.0, 1.0}},
			0.0,
			true,
		},
		{
			"not equal lenghts 2",
			args{a: []float64{1.0, 2.0, 3.0}, b: []float64{2.0, 1.0}},
			0.0,
			true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := scalarMultiplication(tt.args.a, tt.args.b)
			if (err != nil) != tt.wantErr {
				t.Errorf("scalarMultiplication() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !equal(got, tt.want) {
				t.Errorf("scalarMultiplication() = %v, want %v", got, tt.want)
			}
		})
	}
}
