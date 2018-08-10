# Install

```bash
pip install .
```

# Clone the TensorFlow models repository

into some `/path/to/tensorflow/models`:

```bash
git clone git@github.com:tensorflow/models.git
```

# Compile protobuf files

in `/path/to/tensorflow/models/research/`:
```bash
protoc object_detection/protos/*.proto --python_out=.
```

# Set environment variables

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/tensorflow/models/research/
export LABELS_DATA=/path/to/tensorflow/models/research/object_detection/data
```

# Download models

```bash
video detect download-models
```

# Run the detection

```bash
video detect stuff /path/to/video/file
```

See help for options:
```bash
video detect stuff --help
```
