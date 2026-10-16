#!/usr/bin/env python3
"""Generate shared deterministic greedy answers for HAC and CAS.

This is Step 0 of the benchmark. HAC, CAS, and labels must all refer to these
rows rather than generating their own target answers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_common import (  # noqa: E402
    BENCH_ROOT,
    append_jsonl,
    done_indices,
    load_dataset,
    output_path,
    parse_generation,
    sample_id_for,
    sanitize,
    set_seed,
    token_ids_for_answer,
)

from medvlm.confidence import hedge as hedge_mod  # noqa: E402
from medvlm.confidence.sampling import BASE_SAMPLING_PROMPT  # noqa: E402
import medvlm  # noqa: E402


def generate_one(model: Any, processor: Any, inputs: Mapping[str, Any], max_new_tokens: int) -> tuple[str, list[int]]:
    tokenizer = getattr(processor, 'tokenizer', processor)
    pad_id = getattr(tokenizer, 'pad_token_id', None)
    eos_id = getattr(tokenizer, 'eos_token_id', None)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = int(inputs['input_ids'].shape[1])
    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=False,
            num_return_sequences=1,
            temperature=1.0,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_id,
            output_scores=False,
            return_dict_in_generate=True,
        )
    gen_ids_tensor = out.sequences[:, prompt_len:][0]
    gen_ids: list[int] = []
    stop_ids = {int(x) for x in (pad_id, eos_id) if x is not None}
    for token_id in gen_ids_tensor.detach().cpu().tolist():
        token_id = int(token_id)
        if token_id in stop_ids:
            break
        gen_ids.append(token_id)
    text = processor.batch_decode([gen_ids], skip_special_tokens=True)[0] if gen_ids else ''
    return text, gen_ids


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for model_name in args.models:
        print(f'Loading model {model_name}', flush=True)
        model, processor, model_config = medvlm.load_model(
            model_name,
            quantization=args.quantization,
            use_flash_attention=not args.no_flash_attention,
            device_map=args.device_map,
        )
        for dataset in args.datasets:
            for qtype in args.qtypes:
                ds = load_dataset(dataset, qtype, args.seed, args.max_samples)
                out_path = output_path(out_dir, model_name, dataset, qtype)
                done = done_indices(out_path) if args.resume else set()
                max_new_tokens = args.max_new_tokens_closed if qtype == 'closed' else args.max_new_tokens_open
                for idx, sample in enumerate(ds):
                    if idx in done:
                        continue
                    sample_seed = args.seed * 1_000_003 + idx
                    set_seed(sample_seed)
                    try:
                        question = str(sample['question'])
                        inputs = hedge_mod._format_single(
                            processor,
                            model_config.model_family,
                            sample['image'],
                            question,
                            BASE_SAMPLING_PROMPT,
                        )
                        raw_generation, raw_token_ids = generate_one(model, processor, inputs, max_new_tokens)
                        target_answer, target_prediction = parse_generation(raw_generation, qtype)
                        row = {
                            'version': 'shared_greedy_v1',
                            'status': 'ok',
                            'model': model_name,
                            'dataset': dataset,
                            'qtype': qtype,
                            'dataset_index': idx,
                            'sample_id': sample_id_for(idx, sample),
                            'question_id': sample.get('question_id'),
                            'image_id': sample.get('image_id'),
                            'question': question,
                            'ground_truth': sample.get('answer'),
                            'prompt_template': 'BASE_SAMPLING_PROMPT',
                            'raw_generation': raw_generation,
                            'raw_generated_token_ids': raw_token_ids,
                            'target_answer': target_answer,
                            'target_prediction': target_prediction,
                            'target_token_ids': token_ids_for_answer(processor, target_answer),
                            'decode_config': {
                                'do_sample': False,
                                'temperature': None,
                                'max_new_tokens': max_new_tokens,
                                'seed': sample_seed,
                            },
                        }
                    except Exception as exc:
                        row = {
                            'version': 'shared_greedy_v1',
                            'status': 'error',
                            'model': model_name,
                            'dataset': dataset,
                            'qtype': qtype,
                            'dataset_index': idx,
                            'sample_id': sample_id_for(idx, sample),
                            'question_id': sample.get('question_id'),
                            'image_id': sample.get('image_id'),
                            'question': sample.get('question'),
                            'ground_truth': sample.get('answer'),
                            'error': repr(exc),
                        }
                    append_jsonl(out_path, row)
                    print(f'[{model_name}/{dataset}/{qtype}] {idx + 1}/{len(ds)} {row["status"]}', flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--qtypes', nargs='+', choices=['closed', 'open'], required=True)
    parser.add_argument('--output-dir', default=str(BENCH_ROOT / '02_inputs/greedy_answers'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--max-new-tokens-closed', type=int, default=16)
    parser.add_argument('--max-new-tokens-open', type=int, default=64)
    parser.add_argument('--quantization', default=None)
    parser.add_argument('--device-map', default='auto')
    parser.add_argument('--no-flash-attention', action='store_true')
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
