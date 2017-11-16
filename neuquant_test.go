package neuquant

import (
	"bytes"
	"image"
	"image/gif"
	_ "image/jpeg"
	"os"
	"reflect"
	"testing"
)

const (
	imgWidth    = 600
	imgHeight   = 400
	imgFilename = "testdata/testimg.jpg"
)

func TestQuantize(t *testing.T) {
	m := mustReadImg(t, imgFilename)
	q := New()
	p := q.Quantize(nil, m)
	buf := new(bytes.Buffer)
	err := gif.Encode(buf, m, &gif.Options{Quantizer: q, NumColors: 256})
	if err != nil {
		t.Fatalf("Failed to encode image: %v", err)
	}
	gif, err := gif.DecodeAll(buf)
	if err != nil {
		t.Fatalf("Failed to decode gif: %v", err)
	}
	want := gif.Image[0].Palette
	if !reflect.DeepEqual(p, want) {
		t.Errorf("Quantize() = %v, want, %v", p, want)
	}
}

func TestExtractPixels(t *testing.T) {
	m := mustReadImg(t, imgFilename)
	pixels, err := extractPixels(m)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if n, want := len(pixels), imgWidth*imgHeight; n != want {
		t.Errorf("len(pixels) = %d, want %d", n, want)
	}
}

func mustReadImg(t *testing.T, filename string) image.Image {
	t.Helper()
	f, err := os.Open(filename)
	if err != nil {
		t.Fatalf("Failed to open %s: %v", filename, err)
	}
	defer f.Close()
	m, _, err := image.Decode(f)
	if err != nil {
		t.Fatalf("Failed to decode image: %v", err)
	}
	return m
}
