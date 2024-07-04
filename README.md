# go-mnist-experiment

Experimental work & investigation with ML golang libraries.

Model to recognise 28x28 mnist numbers from 0..9.


Train model (remove dump.bin to train new model)
```go
go run main.go train
```

Predict number
```go
go run main.go predict=./data/numbers/mnist_png/Hnd/Sample9/133.png
```