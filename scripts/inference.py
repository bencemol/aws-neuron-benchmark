import logging
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from time import perf_counter
from collections import Counter
import concurrent.futures
import numpy as np
import csv
import torch
import os
import math
# Set up logging
logger = logging.getLogger(__name__)


def get_bucket(start, end, window_size):
  bucketed_start = math.floor(start / window_size) * window_size
  bucketed_end = math.ceil(end / window_size) * window_size
  if bucketed_end - bucketed_start == window_size:
    return bucketed_start
  else:
    return None


def generate_sample_inputs(tokenizer, sequence_length):
  dummy_input = "dummy"
  embeddings = tokenizer(dummy_input, max_length=sequence_length, padding="max_length", return_tensors="pt")
  return tuple(embeddings.values())


def measure_latency_and_throughput(model, tokenizer, sequence_length, n_models = 2, n_threads = 2, batches_per_thread = 100):
  payload = generate_sample_inputs(tokenizer, sequence_length)
  traced_model = torch.jit.trace(model, payload)
  torch.jit.save(traced_model, "models/traced.pt")
  models = [torch.jit.load("models/traced.pt") for _ in range(n_models)]
  times = []
  
  def task(model):
    for _ in range(batches_per_thread):
      start_time = perf_counter()
      _ =  model(*payload)
      end_time = perf_counter()
      times.append((start_time, end_time))


  # warm up
  for _ in range(10):
    for model in models:
      _ = model(*payload)

  # Submit tasks
  with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as pool:
    for i in range(n_threads):
      pool.submit(task, models[i % len(models)])

  # Compute latency
  latencies = [end - start for start, end in times]
  time_avg_ms = 1000 * np.mean(latencies)
  time_std_ms = 1000 * np.std(latencies)
  time_p95_ms = 1000 * np.percentile(latencies,95)

  # Compute throughput
  window_size = 1
  bucketed_timestamps = [get_bucket(start, end, window_size)
                         for start, end in times]
  counted_buckets = Counter(
    item for item in bucketed_timestamps if item is not None)
  bucket_throughput = [(key, value / window_size)
                        for key, value in sorted(counted_buckets.items())]
  busy_throughput = [value for _, value in bucket_throughput]
  max_throughput = max(busy_throughput) * sequence_length
  avg_throughput = sum(busy_throughput) * sequence_length / len(busy_throughput)

  return {"throughput_max": max_throughput, "throughput_avg": avg_throughput, "time_avg_ms": time_avg_ms, "time_std_ms": time_std_ms, "time_p95_ms": time_p95_ms, "sequence_length": sequence_length}


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--is_neuron", action="store_true")
  parser.add_argument("--model_id", type=str)  
  parser.add_argument("--instance_type", type=str)  
  parser.add_argument("--sequence_length", nargs="+", type=int, default=None)
  
  # neuron specific args
  parser.add_argument("--num_neuron_cores", type=int, default=1)
  parser.add_argument("--num_threads", type=int, default=1)
  known_args, _ = parser.parse_known_args()  
  return known_args

def compile_model_inf1(model, tokenizer, sequence_length, num_neuron_cores):
  os.environ['NEURON_RT_NUM_CORES'] = str(num_neuron_cores)
  import torch.neuron
  payload = generate_sample_inputs(tokenizer, sequence_length)
  return torch.neuron.trace(model, payload)

def compile_model_inf2(model, tokenizer, sequence_length, num_neuron_cores):
  # use only one neuron core
  os.environ["NEURON_RT_NUM_CORES"] = str(num_neuron_cores)
  import torch_neuronx
  payload = generate_sample_inputs(tokenizer, sequence_length)
  return torch_neuronx.trace(model, payload)

def write_to_csv(dict, instance_type, model_id, prefix = ""):
  keys = dict[0].keys()
  if prefix:
    prefix += "_"
  with open(f'results/{prefix}{instance_type}_{model_id.replace("-","_").replace("/","_")}.csv', 'w', newline='') as output_file:
      dict_writer = csv.DictWriter(output_file, keys)
      dict_writer.writeheader()
      dict_writer.writerows(dict)


def main(args):
  print(args)

  # define sequence lengths to benchmark
  if args.sequence_length is None:
    # sequence_lengths = [8,16,32,64,128, 256, 512] 
    sequence_lengths = [512]
  else:
    sequence_lengths = args.sequence_length

  # benchmark model
  result_dict = []
  for sequence_length in sequence_lengths:
    # load tokenizer and  model
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_id, torchscript=True)
    
 
    # compile model if neuron
    if args.is_neuron:
      if "inf1" in args.instance_type:
        model = compile_model_inf1(model, tokenizer, sequence_length, args.num_neuron_cores)
      elif "inf2" in args.instance_type:
        model = compile_model_inf2(model, tokenizer, sequence_length, args.num_neuron_cores)
      else:
        raise ValueError("Unknown neuron version")

    logger.info(f"Measuring latency and throughput for sequence length {sequence_length}")
    res = measure_latency_and_throughput(model, tokenizer, sequence_length, args.num_neuron_cores, args.num_threads)
    print(res)
    result_dict.append({**res,"instance_type": args.instance_type})
    
  # write results to csv
  write_to_csv(result_dict, args.instance_type, args.model_id)


if __name__ == "__main__":
  main(parse_args())
