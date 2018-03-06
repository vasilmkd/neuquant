/*
NeuQuant Nerual-Net Quantization Algorithm
------------------------------------------

Copyright (c) 1994 Anthony Dekker

NEUQUANT Nerual-Net quantization algorithm by Anthony Dekker, 1994.
See "Kohonen neural networks for optimal colour quantization"
in "Network: Computation in Neural Systems" Vol. 5 (1994) pp 351-367.
for a discussion of the algorithm.
See also http://www.acm.org/~dekker/NEUQUANT.HTML

Any party obtaining a copy of these files from the author, directly or
indirectly, is granted, free of charge, a full and unrestricted irrevocable,
world-wide, paid up, royalty-free, nonexclusive right and licence to deal
in this software and documentation files (the "Software"), including without
limitation the rights to use, copy, modify, merge, publish, distribute, sublicence,
and/or sell copies of the Software, and to permit persons who receive
copies from any such party to do so, with the only requirement being
that this copyright notice remain intact.
*/

// Package neuquant implements the NeuQuant Neural-Net quantization algorithm
// by Anthony Dekker.
package neuquant

import (
	"errors"
	"image"
	"image/color"
	"image/draw"
	"math"
)

const (
	numCycles = 100 // Number of learning cycles.

	netSize  = 256 // Number of colors used.
	specials = 3   // Number of reserved colors used.

	initRad      = netSize / 8 // For 256 colors, radius starts at 32.
	radBiasShift = 6
	radBias      = 1 << radBiasShift
	initBiasRad  = initRad * radBias
	radDec       = 30 // Factor of 1/30 each cycle.

	alphaBiasShift = 10
	initAlpha      = 1 << alphaBiasShift // Alpha starts at 1 biased by 10 bits.

	beta  = 1.0 / 1024.0
	gamma = 1024.0

	// Four primes near 500 - assume no image has a length so large
	// that it is divisible by all four primes.
	prime1 = 499
	prime2 = 491
	prime3 = 487
	prime4 = 503
)

// Quantizer is a Kohonen neural network color quantizer, used to
// quantize an image into, at most, 256 distinct colors. It implements the
// draw.Quantizer interface. Useful for encoding images in the GIF image format.
type quantizer struct {
	sampleFac int

	network  [netSize][3]float64 // The network itself.
	colorMap [netSize][3]int

	netIndex [256]int // For network lookup.

	bias, freq [netSize]float64

	pixels []int
}

// New returns a new Kohonen neural network color quantizer with a sampling
// factor of 1 (best quality).
func New() draw.Quantizer {
	return NewWithSamplingFactor(1)
}

// NewWithSamplingFactor returns a new Kohonen neural network color quantizer
// with the specified sampling factor. The sampling factor must be in the
// range [1,30]. Higher numbers reduce computation time at the expense of
// image quality.
func NewWithSamplingFactor(sample int) draw.Quantizer {
	if sample < 1 || sample > 30 {
		panic("sample must be between 1 and 30")
	}
	return &quantizer{sampleFac: sample}
}

// Quantize creates a color palette suitable for converting m to a palleted
// image.
func (q *quantizer) Quantize(p color.Palette, m image.Image) color.Palette {
	q.setPixels(m)
	q.setUpArrays()
	q.learn()
	q.fix()
	q.inxBuild()
	return makePalette(q.colorMap)
}

func (q *quantizer) setPixels(im image.Image) {
	pixels, err := extractPixels(im)
	if err != nil {
		panic(err)
	}
	q.pixels = pixels[:]
}

func (q *quantizer) setUpArrays() {
	q.network[0][0] = 0.0 // Black.
	q.network[0][1] = 0.0
	q.network[0][2] = 0.0

	q.network[1][0] = 255.0 // White.
	q.network[1][1] = 255.0
	q.network[1][2] = 255.0

	for i := 0; i < specials; i++ {
		q.freq[i] = 1.0 / float64(netSize)
		q.bias[i] = 0.0
	}

	cutNetSize := netSize - specials
	for i := specials; i < netSize; i++ {
		p := q.network[i][:]
		p[0] = (255.0 * float64(i-specials)) / float64(cutNetSize)
		p[1] = (255.0 * float64(i-specials)) / float64(cutNetSize)
		p[2] = (255.0 * float64(i-specials)) / float64(cutNetSize)

		q.freq[i] = 1.0 / float64(netSize)
		q.bias[i] = 0.0
	}
}

func (q *quantizer) learn() {
	biasRad := initBiasRad
	alphaDec := 30 + ((q.sampleFac - 1) / 3)
	lengthCount := len(q.pixels)
	samplePixels := lengthCount / q.sampleFac
	delta := samplePixels / numCycles
	alpha := initAlpha

	rad := calcRad(biasRad)

	step, pos := 0, 0

	if lengthCount%prime1 != 0 {
		step = prime1
	} else {
		if lengthCount%prime2 != 0 {
			step = prime2
		} else {
			if lengthCount%prime3 != 0 {
				step = prime3
			} else {
				step = prime4
			}
		}
	}

	i := 0
	for i < samplePixels {
		p := q.pixels[pos]
		red := uint32((p >> 16) & 0xFF)
		green := uint32((p >> 8) & 0xFF)
		blue := uint32(p & 0xFF)

		r, g, b := float64(red), float64(green), float64(blue)

		// Remember background color.
		if bgColor := specials - 1; i == 0 {
			q.network[bgColor][0] = r
			q.network[bgColor][1] = g
			q.network[bgColor][2] = b
		}

		j := q.specialFind(r, g, b)
		if j < 0 {
			j = q.contest(r, g, b)
		}

		// Don't learn for specials.
		if j >= specials {
			a := float64(alpha) / float64(initAlpha)
			q.alterSingle(a, j, r, g, b)
			if rad > 0 {
				q.alterNeighbors(a, rad, j, r, g, b)
			}
		}

		pos += step
		pos = pos % lengthCount

		i++
		if i%delta == 0 {
			alpha -= alpha / alphaDec
			biasRad -= biasRad / radDec
			rad = calcRad(biasRad)
		}
	}
}

func (q *quantizer) fix() {
	for i := 0; i < netSize; i++ {
		for j := 0; j < 3; j++ {
			q.colorMap[i][j] = roundToColorValue(q.network[i][j])
		}
	}
}

// Insertion sort of network and building of netIndex[0..255]
func (q *quantizer) inxBuild() {
	maxNetPos := netSize - 1
	prevCol, startPos := 0, 0

	for i := 0; i < netSize; i++ {
		p := q.colorMap[i][:]
		smallPos, smallVal := i, p[1]

		for j := i + 1; j < netSize; j++ {
			c := q.colorMap[j][:]
			if c[1] < smallVal {
				smallPos = j
				smallVal = c[1]
			}
		}

		c := q.colorMap[smallPos][:]

		if i != smallPos {
			p[0], c[0] = c[0], p[0]
			p[1], c[1] = c[1], p[1]
			p[2], c[2] = c[2], p[2]
		}

		if smallVal != prevCol {
			q.netIndex[prevCol] = (startPos + i) >> 1
			for j := prevCol + 1; j < smallVal; j++ {
				q.netIndex[j] = i
			}
			prevCol = smallVal
			startPos = i
		}
	}
	q.netIndex[prevCol] = (startPos + maxNetPos) >> 1
	for j := prevCol + 1; j < 256; j++ {
		q.netIndex[j] = maxNetPos
	}
}

// Move neuron i towards (r, g, b).
func (q *quantizer) alterSingle(alpha float64, i int, r, g, b float64) {
	p := q.network[i][:]
	p[0] -= alpha * (p[0] - r)
	p[1] -= alpha * (p[1] - g)
	p[2] -= alpha * (p[2] - b)
}

// Move all neurons that are at most rad away from i towards (r, g, b).
func (q *quantizer) alterNeighbors(alpha float64, rad int, i int, r, g, b float64) {
	lo, hi := i-rad, i+rad
	if lo < specials {
		lo = specials - 1
	}
	if hi > netSize {
		hi = netSize
	}

	j, k := i+1, i-1
	var c int
	for (j < hi) || (k > lo) {
		a := (alpha * float64(rad*rad-c*c)) / float64(rad*rad)
		c++
		if j < hi {
			p := q.network[j][:]
			p[0] -= a * (p[0] - r)
			p[1] -= a * (p[1] - g)
			p[2] -= a * (p[2] - b)
			j++
		}
		if k > lo {
			p := q.network[k][:]
			p[0] -= a * (p[0] - r)
			p[1] -= a * (p[1] - g)
			p[2] -= a * (p[2] - b)
			k--
		}
	}
}

// Searches for biased RGB values, finds the closest neuron (minimum distance)
// and updates freq, finds the best neuron (minimum dist - bias) and returns
// its position. For frequently chosen neurons, freq[i] is high and bias[i] is
// negative.
// bias[i] = gamma * ((1 / netSize) - freq[i])
func (q *quantizer) contest(r, g, b float64) int {
	bestDist := math.MaxFloat64
	bestBiasDist := bestDist
	bestPos := -1
	bestBiasPos := bestPos

	for i := specials; i < netSize; i++ {
		p := q.network[i][:]
		dist := math.Abs(p[0]-r) + math.Abs(p[1]-g) + math.Abs(p[2]-b)
		if dist < bestDist {
			bestDist = dist
			bestPos = i
		}
		biasDist := dist - q.bias[i]
		if biasDist < bestBiasDist {
			bestBiasDist = biasDist
			bestBiasPos = i
		}
		q.freq[i] -= beta * q.freq[i]
		q.bias[i] += beta * gamma * q.freq[i]
	}
	q.freq[bestPos] += beta
	q.bias[bestPos] -= beta * gamma
	return bestBiasPos
}

func (q *quantizer) specialFind(r, g, b float64) int {
	for i := 0; i < specials; i++ {
		p := q.network[i][:]
		if eqFloat(p[0], r) && eqFloat(p[1], g) && eqFloat(p[2], b) {
			return i
		}
	}
	return -1
}

func extractPixels(m image.Image) ([]int, error) {
	w := m.Bounds().Max.X
	h := m.Bounds().Max.Y
	if w*h < prime4 {
		return nil, errors.New("image is too small")
	}
	var pixels []int
	for y := m.Bounds().Min.Y; y < h; y++ {
		for x := m.Bounds().Min.X; x < w; x++ {
			r, g, b, _ := m.At(x, y).RGBA()
			px := int((r << 16) | (g << 8) | b)
			pixels = append(pixels, px)
		}
	}
	return pixels, nil
}

func calcRad(bias int) int {
	rad := bias >> radBiasShift
	if rad <= 1 {
		rad = 0
	}
	return rad
}

func eqFloat(a, b float64) bool {
	return math.Abs(a-b) < 1e-5
}

func roundToColorValue(x float64) int {
	res := int(0.5 + x)
	if res < 0 {
		res = 0
	} else if res > 255 {
		res = 255
	}
	return res
}

func makePalette(colorMap [netSize][3]int) color.Palette {
	var res color.Palette
	for i := 0; i < netSize; i++ {
		c := color.RGBA{
			R: uint8(colorMap[i][0]),
			G: uint8(colorMap[i][1]),
			B: uint8(colorMap[i][2]),
			A: 0xFF,
		}
		res = append(res, c)
	}
	return res
}
