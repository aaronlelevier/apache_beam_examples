import apache_beam as beam
from apache_beam.io import ReadFromText, WriteToText
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions

pipeline_options = PipelineOptions()
pipeline_options.view_as(SetupOptions).save_main_session = True

with beam.Pipeline(options=pipeline_options) as p:

    infile = '../notes/wordcount-example.md'
    outfile = '../notes/wordcount-example-OUT-2.md'

    lines = p | beam.io.ReadFromText(infile)

    out_lines = lines | beam.io.WriteToText(outfile)
