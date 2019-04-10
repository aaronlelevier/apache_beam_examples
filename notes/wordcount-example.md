# Learning Beam Notes

Notes for [wordcount-example](https://beam.apache.org/get-started/wordcount-example/), the "hello world" of data processing

First construct a Pipeline with `beam.Pipeline`

re-running a Pipeline from a middle step doesn't work because Transforms are mutating the Pipeline

Subclass `beam.PTransform` to create composite transforms