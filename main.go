package main

import (
	"encoding/binary"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"github.com/patrikeh/go-deep"
	"github.com/patrikeh/go-deep/training"
	"image"
	_ "image/png"
	"log"
	"os"
	"strconv"
	"strings"
)

const (
	iterations = 5
)

const (
	csvHeaderOrigin = iota
	csvHeaderGroup
	csvHeaderLabel
	csvHeaderFile
)

const (
	appArgLearn   = "train"
	appArgPredict = "predict"
)

const img28x28 = 28

func main() {
	args := argsToMap()

	if _, ok := args.Get(appArgLearn); ok {
		file, err := os.Open("./data/numbers.csv")
		if err != nil {
			panic(err)
		}
		reader := csv.NewReader(file)
		records, err := reader.ReadAll()
		if err != nil {
			panic(err)
		}

		imageResult := make(map[string]int64)

		for _, r := range records {
			if r[csvHeaderOrigin] == "mnist" {
				parseInt, err := strconv.ParseInt(r[csvHeaderLabel], 10, 64)
				if err != nil {
					panic(err)
				}

				imageResult["./data/numbers/"+r[csvHeaderFile]] = parseInt
			}
		}

		transformedImages := getTransformedImage(imageResult)

		train(transformedImages)
	}

	if filePath, ok := args.Get(appArgPredict); ok {
		imageResult := make(map[string]int64)
		imageResult[filePath] = 0 // for now zero, should be refactored
		transformedImages := getTransformedImage(imageResult)
		data := transformForNeuro(transformedImages)

		neuro := loadNeuroFromDump()

		result := neuro.Predict(data[0].Input)
		output := indexOfMax(result)
		log.Printf("ResPred: %+v", result)
		println(output)
	}
}

func indexOfMax(arr []float64) int {
	maxIdx := 0
	maxVal := arr[0]
	for i := 0; i < len(arr)-1; i++ {
		if arr[i+1] > maxVal {
			maxIdx = i + 1
			maxVal = arr[i+1]
		}
	}
	return maxIdx
}

type Args map[string]bool

func (a Args) Get(argKey string) (string, bool) {
	if len(a) == 0 {
		return "", false
	}
	for k := range a {
		if argKey == k {
			return "", true
		}
		if strings.Contains(k, argKey) {
			return strings.Split(k, "=")[1], true
		}
	}
	return "", false
}

func argsToMap() Args {
	args := make(Args)
	for _, v := range os.Args {
		args[v] = true
	}
	return args
}

type TransformedImagesWithResult struct {
	Images   []float64
	Value    []float64
	FilePath string
}

func getTransformedImage(imageResult map[string]int64) []TransformedImagesWithResult {
	var transformedImages []TransformedImagesWithResult

	for filePath, value := range imageResult {
		valuesArr := []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
		var imgArr []float64

		readFile, err := os.Open(filePath)
		if err != nil {
			panic(err)
		}

		img, _, err := image.Decode(readFile)
		if err != nil {
			panic(err)
		}

		for x := 0; x < img28x28; x++ {
			for y := 0; y < img28x28; y++ {
				im := img.At(x, y)
				r, g, b, a := im.RGBA()
				alpha := float64(a) / float64(255)
				if r == 0 && b == 0 && g == 0 {
					imgArr = append(imgArr, 0)
					continue
				}
				pixel := ((float64(r)/float64(255) + float64(g)/float64(255) + float64(b)/float64(255)) / float64(3)) / alpha
				imgArr = append(imgArr, pixel)
			}
		}

		valuesArr[value] = 1

		transformedImages = append(transformedImages, TransformedImagesWithResult{
			Images:   imgArr,
			Value:    valuesArr,
			FilePath: filePath,
		})
	}
	return transformedImages
}

func train(transformedImages []TransformedImagesWithResult) {
	data := transformForNeuro(transformedImages)
	test := transformForNeuro(transformedImages)

	n := loadNeuroFromDump()

	if n == nil {
		n = deep.NewNeural(&deep.Config{
			/* Input dimensionality */
			Inputs: 784,
			/* Two hidden layers consisting of two neurons each, and a single output */
			Layout: []int{512, 512, 10},
			/* Activation functions: Sigmoid, Tanh, ReLU, Linear */
			Activation: deep.ActivationSigmoid,
			/* Determines output layer activation & loss function:
			ModeRegression: linear outputs with MSE loss
			ModeMultiClass: softmax output with Cross Entropy loss
			ModeMultiLabel: sigmoid output with Cross Entropy loss
			ModeBinary: sigmoid output with binary CE loss */
			Mode: deep.ModeBinary,
			/* Weight initializers: {deep.NewNormal(μ, σ), deep.NewUniform(μ, σ)} */
			Weight: deep.NewUniform(1.0, 0.0),
			/* Apply bias */
			Bias: true,
		})
	}

	log.Printf("Numbers of Weights: %d", n.NumWeights())

	defer func() {
		dumpData, err := n.Marshal()
		if err != nil {
			panic(err)
		}
		dump(dumpData)
	}()

	// params: learning rate, momentum, alpha decay, nesterov
	optimizer := training.NewSGD(0.001, 0.9, 1e-6, true)
	// params: optimizer, verbosity (print stats at every 50th iteration)
	trainer := training.NewBatchTrainer(optimizer, 1, 128, 3)

	trainer.Train(n, data, test, iterations) // training, validation, iterations

	fmt.Println(transformedImages[0].FilePath, transformedImages[0].Value, "=>", n.Predict(data[0].Input))
	fmt.Println(transformedImages[5].FilePath, transformedImages[5].Value, "=>", n.Predict(data[5].Input))
}

func transformForNeuro(transformedImages []TransformedImagesWithResult) training.Examples {
	var data training.Examples
	for _, transformedImage := range transformedImages {
		data = append(data, training.Example{Input: transformedImage.Images, Response: transformedImage.Value})
	}
	return data
}

func loadNeuroFromDump() *deep.Neural {
	var n *deep.Neural
	dumpFile, err := os.ReadFile("./dump.bin")
	if err != nil {
		println("Dump file can't be read :(")
		println(err)
	} else {
		unmarshal, err := Unmarshal(dumpFile)
		if err != nil {
			println("Can't unmarshal dump file", err)
		} else {
			println("Neural used from dump file.")
			n = unmarshal
		}
	}
	return n
}

func dump(data any) {
	println("dumping...")

	f, err := os.Create("dump.bin")
	if err != nil {
		log.Fatal("Couldn't open file")
	}
	defer f.Close()

	err = binary.Write(f, binary.BigEndian, data)
	if err != nil {
		fmt.Println(err)
	}
}

func Unmarshal(bytes []byte) (*deep.Neural, error) {
	var dump deep.Dump
	if err := json.Unmarshal(bytes, &dump); err != nil {
		return nil, err
	}
	return deep.FromDump(&dump), nil
}
