package regression

import (
	"fmt"
	"strings"
)

type History struct {
	items []historyItem
}

type historyItem struct {
	iteration int
	loss      float64
	weights   []float64
	gradient  []float64
}

func (h *History) Push(iteration int, loss float64, weights, gradient []float64) {
	h.items = append(h.items, historyItem{
		iteration,
		loss,
		weights,
		gradient,
	})
}

func (h *History) String() string {
	result := make([]string, 0)

	for _, item := range h.items {
		itemMessage := fmt.Sprintf("iteration [%d], loss: %.5f, weights: %v, gradients: %v", item.iteration, item.loss, item.weights, item.gradient)
		result = append(result, itemMessage)
	}

	return strings.Join(result, "\n")
}
