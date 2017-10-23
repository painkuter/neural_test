package main

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestGetRandWeights(t *testing.T) {
	assert.Equal(t, len(GetRandWeights(6)), 6)
}

func TestRound(t *testing.T) {
	assert.Equal(t, Round(0.12345, .5, 2), 0.12, "Round")
}

func Test(t *testing.T) {
	main()
}
