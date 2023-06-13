# Performance Comparison of Pretrained Huggingface Models on AWS GPU and Inferentia Instances

## Getting Started

Install the required dependencies by running:

```python
pip install -r requirements.txt
```

## Compiling a Model with Neuron

```sh
python scripts/compile.py --model_id SZTAKI-HLT/hubert-base-cc --is_neuron --num_neuron_cores 4 --instance_type inf1.xlarge --batch_size 6 7 8 9 10
```

## Running the Inference Benchmark

```sh
python scripts/inference.py --model_id SZTAKI-HLT/hubert-base-cc --is_neuron --num_neuron_cores 2 --num_threads 2 --batch_size 1 2 3 4 5 6 7 8 9 10 --instance_type inf2.xlarge
```
