package datasets

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
)

var irisTargets = map[string]float64{
	"setosa":     0.0,
	"versicolor": 1.0,
	"virginica":  2.0,
}

func LoadIris() ([][]float64, []float64, error) {
	file, err := os.Open("datasets/iris.csv")
	if err != nil {
		return nil, nil, fmt.Errorf("Couldn't read iris dataset: %v", err)
	}

	records, err := csv.NewReader(file).ReadAll()
	if err != nil {
		return nil, nil, fmt.Errorf("Couldn't read iris dataset: %v", err)
	}

	objects := make([][]float64, 0)
	targets := make([]float64, 0)

	for _, record := range records[1:] {
		object := make([]float64, 0)
		for _, featureStringValue := range record[0:3] {
			value, err := strconv.ParseFloat(featureStringValue, 64)
			if err != nil {
				return nil, nil, fmt.Errorf("Couldn't read iris dataset: %v", err)
			}
			object = append(object, value)
		}

		targetValue, err := encodeTarget(record[4])
		if err != nil {
			return nil, nil, fmt.Errorf("Couldn't read iris dataset: %v", err)
		}

		objects = append(objects, object)
		targets = append(targets, targetValue)
	}

	return objects, targets, nil

}

func encodeTarget(target string) (float64, error) {
	item, ok := irisTargets[target]
	if !ok {
		return -1.0, fmt.Errorf("wrong iris target value: %s", target)
	}

	return item, nil
}
