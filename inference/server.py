"""Single-request Transformers VLM service for EmbodiedBench (no LMDeploy)."""
import argparse
import importlib.metadata
import json
import threading
import time
import uuid
from pathlib import Path

import torch
import transformers
import xgrammar as xgr
from fastapi import FastAPI, HTTPException
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from xgrammar.contrib.hf import LogitsProcessor

app = FastAPI(title="vlm-planning Transformers service")
lock = threading.Lock()
model = processor = args = grammar_compiler = None


@app.get('/health')
def health():
    return dict(status='ok' if model is not None else 'loading',
                model_path=args.model_path, model_name=args.model_name,
                transformers_source=transformers.__file__,
                transformers_version=transformers.__version__,
                xgrammar_version=importlib.metadata.version('xgrammar'),
                dtype=args.dtype, attention='sdpa',
                schema_mode='xgrammar+jsonschema-validation')


@app.get('/v1/models')
def models():
    return {'object': 'list', 'data': [{'id': args.model_name, 'object': 'model',
                                      'created': 0, 'owned_by': 'local'}]}


def convert_messages(messages):
    """Convert OpenAI image parts without changing the original conversation."""
    converted = []
    for message in messages:
        content = message.get('content', '')
        if isinstance(content, list):
            parts = []
            for item in content:
                if item['type'] == 'text':
                    parts.append({'type': 'text', 'text': item['text']})
                elif item['type'] == 'image_url':
                    url = item['image_url']
                    url = url['url'] if isinstance(url, dict) else url
                    if not url.startswith('data:image/'):
                        raise HTTPException(400, 'Only inline data:image URLs are supported')
                    parts.append({'type': 'image', 'image': url})
                else:
                    raise HTTPException(400, 'Unsupported content type')
            content = parts
        converted.append({'role': message['role'], 'content': content})
    return converted


def prepare_response_constraint(response_format):
    """Build a fresh stateful logits processor and a post-generation validator."""
    kind = response_format.get('type', 'text')
    if kind == 'text':
        return None, None
    if kind == 'json_schema':
        schema = response_format.get('json_schema', {}).get('schema')
        if not isinstance(schema, dict):
            raise HTTPException(400, 'json_schema.schema must be an object')
        try:
            Draft202012Validator.check_schema(schema)
            compiled = grammar_compiler.compile_json_schema(schema, strict_mode=True)
        except (SchemaError, RuntimeError, ValueError) as error:
            raise HTTPException(400, f'Invalid or unsupported JSON schema: {error}') from error
        validator = Draft202012Validator(schema)
    elif kind == 'json_object':
        schema = {'type': 'object'}
        compiled = grammar_compiler.compile_json_schema(schema, strict_mode=True)
        validator = Draft202012Validator(schema)
    else:
        raise HTTPException(400, 'Unsupported response_format')
    return LogitsProcessor(compiled), validator


def validate_structured_output(text, validator):
    if validator is None:
        return
    try:
        value = json.loads(text)
        validator.validate(value)
    except json.JSONDecodeError as error:
        raise HTTPException(500, f'Constrained decoding produced invalid JSON: {error}',
                            headers={'x-should-retry': 'false'}) from error
    except ValidationError as error:
        raise HTTPException(500, f'Model output failed JSON schema validation: {error.message}',
                            headers={'x-should-retry': 'false'}) from error


@app.post('/v1/chat/completions')
def complete(request: dict):
    if request.get('model') != args.model_name:
        raise HTTPException(404, 'Model name does not match /v1/models')
    if request.get('stream') or request.get('n', 1) != 1:
        raise HTTPException(400, 'Only non-streaming n=1 requests are supported')
    messages = request.get('messages', [])
    if not messages:
        raise HTTPException(400, 'messages must not be empty')
    limit = int(request.get('max_completion_tokens', request.get('max_tokens', 2048)))
    if not 1 <= limit <= 4096:
        raise HTTPException(400, 'Completion limit must be 1..4096')
    temperature = float(request.get('temperature', 0))
    if temperature < 0:
        raise HTTPException(400, 'temperature must be nonnegative')
    response_format = request.get('response_format') or {'type': 'text'}
    messages = convert_messages(messages)
    with lock:
        started = time.perf_counter()
        logits_processor, validator = prepare_response_constraint(response_format)
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors='pt').to(model.device)
        prompt_tokens = inputs.input_ids.shape[-1]
        if prompt_tokens + limit > args.max_context:
            raise HTTPException(400, 'Prompt plus completion exceeds configured context limit')
        torch.cuda.synchronize()
        prepared = time.perf_counter()
        kwargs = dict(max_new_tokens=limit, do_sample=temperature > 0)
        if temperature > 0:
            kwargs['temperature'] = temperature
        if logits_processor is not None:
            kwargs['logits_processor'] = [logits_processor]
        with torch.inference_mode():
            output = model.generate(**inputs, **kwargs)
        torch.cuda.synchronize()
        finished = time.perf_counter()
        generated = output[0, prompt_tokens:]
        token_ids = generated.tolist()
        text = processor.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        validate_structured_output(text, validator)
        eos = model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        reason = 'stop' if token_ids and token_ids[-1] in eos else ('length' if len(token_ids) >= limit else 'stop')
    result = dict(id='chatcmpl-' + uuid.uuid4().hex, object='chat.completion', created=int(time.time()),
                  model=args.model_name,
                  choices=[dict(index=0, message=dict(role='assistant', content=text), finish_reason=reason)],
                  usage=dict(prompt_tokens=prompt_tokens, completion_tokens=len(token_ids),
                             total_tokens=prompt_tokens + len(token_ids)),
                  server_timing=dict(preprocessing_seconds=prepared-started, generation_seconds=finished-prepared),
                  schema_mode='xgrammar+jsonschema-validation')
    print(json.dumps({'request_id': result['id'], 'usage': result['usage'],
                      'finish_reason': reason, 'timing': result['server_timing']}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--model-name', default='Qwen/Qwen3-VL-8B-Instruct')
    parser.add_argument('--port', type=int, default=8001)
    parser.add_argument('--dtype', choices=['bfloat16', 'float16'], default='bfloat16')
    parser.add_argument('--max-context', type=int, default=16000)
    args = parser.parse_args()
    args.model_path = str(Path(args.model_path).resolve())
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, local_files_only=True, dtype=getattr(torch, args.dtype),
        device_map={'': 0}, attn_implementation='sdpa').eval()
    vocab_size = model.get_output_embeddings().weight.shape[0]
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        processor.tokenizer, vocab_size=vocab_size)
    grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=args.port)
