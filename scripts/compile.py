import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def generate_sample_inputs(tokenizer, batch_size, max_length=128, is_gpu=False):
  input = "dummy"
  tokens = tokenizer.encode_plus(
        input,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
  )
  return (
      torch.repeat_interleave(tokens['input_ids'], batch_size, 0),
      torch.repeat_interleave(tokens['attention_mask'], batch_size, 0),
      torch.repeat_interleave(tokens['token_type_ids'], batch_size, 0),
  )

def compile_model_inf1(model, tokenizer, batch_size, num_neuron_cores):
  os.environ['NEURON_RT_NUM_CORES'] = str(num_neuron_cores)
  import torch.neuron
  payload = generate_sample_inputs(tokenizer, batch_size)
  return torch.neuron.trace(model, payload)

def compile_model_inf2(model, tokenizer, batch_size, num_neuron_cores):
  # use only one neuron core
  os.environ["NEURON_RT_NUM_CORES"] = str(num_neuron_cores)
  import torch_neuronx
  payload = generate_sample_inputs(tokenizer, batch_size)
  return torch_neuronx.trace(model, payload)

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_id", type=str)
  parser.add_argument("--instance_type", type=str)
  parser.add_argument("--batch_size", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
  known_args, _ = parser.parse_known_args()
  return known_args

def main(args):
  print(args)

  tokenizer = AutoTokenizer.from_pretrained(args.model_id)
  model = AutoModelForSequenceClassification.from_pretrained(args.model_id, torchscript=True)
  
  for batch_size in args.batch_sizes:
    print(f"Compiling model for batch size: {batch_size}...")
    if "inf1" in args.instance_type:
        model = compile_model_inf1(model, tokenizer, batch_size, args.num_neuron_cores)
    elif "inf2" in args.instance_type:
        model = compile_model_inf2(model, tokenizer, batch_size, args.num_neuron_cores)
    else:
        raise ValueError("Unknown neuron version")
    
    model_name = args.model_id.replace("/", "_")
    torch.jit.save(model, f"models/{model_name}_{batch_size}.pt")


if __name__ == "__main__":
  main(parse_args())
