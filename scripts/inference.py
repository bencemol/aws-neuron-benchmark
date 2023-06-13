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
import gc
# Set up logging
logger = logging.getLogger(__name__)


def get_bucket(start, end, window_size):
  bucketed_start = math.floor(start / window_size) * window_size
  bucketed_end = math.ceil(end / window_size) * window_size
  if bucketed_end - bucketed_start == window_size:
    return bucketed_start
  else:
    return None


def generate_sample_inputs(tokenizer, batch_size, max_length=128, is_gpu=False):
  input = "dummy"
  tokens = tokenizer.encode_plus(
        input,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
  )
  if is_gpu:
    embeddings = {k: v.to("cuda") for k, v in embeddings.items()}
    return tuple(embeddings.values())
  return (
      torch.repeat_interleave(tokens['input_ids'], batch_size, 0),
      torch.repeat_interleave(tokens['attention_mask'], batch_size, 0),
      torch.repeat_interleave(tokens['token_type_ids'], batch_size, 0),
  )


def measure_latency_and_throughput(
    model, 
    tokenizer, 
    batch_size,
    n_models = 2, 
    n_threads = 2,
    is_gpu = False,
    batches_per_thread = 10000):
  payload = generate_sample_inputs(tokenizer, batch_size, is_gpu=is_gpu)
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
  max_throughput = max(busy_throughput) * batch_size
  avg_throughput = sum(busy_throughput) * batch_size / len(busy_throughput)

  return {"throughput_max": max_throughput, "throughput_avg": avg_throughput, "time_avg_ms": time_avg_ms, "time_std_ms": time_std_ms, "time_p95_ms": time_p95_ms, "batch_size": batch_size}


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--is_neuron", action="store_true")
  parser.add_argument("--is_gpu", action="store_true")
  parser.add_argument("--model_id", type=str)  
  parser.add_argument("--instance_type", type=str)  
  parser.add_argument("--batch_size", nargs="+", type=int, default=None)
  parser.add_argument("--batches_per_thread", type=int, default=10000)
  
  # neuron specific args
  parser.add_argument("--num_neuron_cores", type=int, default=1)
  parser.add_argument("--num_threads", type=int, default=1)
  known_args, _ = parser.parse_known_args()  
  return known_args


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
  if args.batch_size is None:
    # batch_sizes = [8,16,32,64,128, 256, 512] 
    batch_sizes = [512]
  else:
    batch_sizes = args.batch_size

  tokenizer = AutoTokenizer.from_pretrained(args.model_id)
  model = AutoModelForSequenceClassification.from_pretrained(args.model_id, torchscript=True)
  
  # benchmark model
  result_dict = []
  for batch_size in batch_sizes:
    # load tokenizer and  model
    if args.is_gpu:
      model.to("cuda")
 
    # compile model if neuron
    if args.is_neuron:
      model_name = args.model_id.replace("/", "_")
      compiled_model = f"models/{model_name}_{batch_size}.pt"
      if "inf1" in args.instance_type:
        import torch.neuron
      elif "inf2" in args.instance_type:
        import torch_neuronx
      else:
        raise ValueError("Unknown neuron version")
      model = torch.jit.load(compiled_model)

    logger.info(f"Measuring latency and throughput for batch size {batch_size}")
    res = measure_latency_and_throughput(
      model, 
      tokenizer, 
      batch_size, 
      args.num_neuron_cores, 
      args.num_threads, 
      args.is_gpu, 
      args.batches_per_thread)
    print(res)
    result_dict.append({**res,"instance_type": args.instance_type})
    gc.collect()
    if args.is_gpu:
      torch.cuda.empty_cache()
    
  # write results to csv
  write_to_csv(result_dict, args.instance_type, args.model_id)


if __name__ == "__main__":
  main(parse_args())
