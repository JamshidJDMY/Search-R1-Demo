#!/usr/bin/env python3
"""Local Search-R1 web demo. Run inside the searchr1 conda environment."""
import argparse
import json
import os
from pathlib import Path
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = Path(__file__).resolve().parent / 'web_demo' / 'index.html'
MODEL_ID = os.environ.get('SEARCHR1_MODEL_ID', 'PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo')
RETRIEVER_URL = os.environ.get('RETRIEVER_URL', 'http://127.0.0.1:8000').rstrip('/')


class Inference:
    def __init__(self):
        self.lock = threading.Lock()
        self.model = None

    def load(self):
        if self.model is not None:
            return
        print(f'Loading model: {MODEL_ID}. Startup can take several minutes…', flush=True)
        import torch
        import transformers
        self.torch = torch
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, device_map='auto')
        model.eval()
        tokenizer = self.tokenizer

        class StopOnSearch(transformers.StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                # Match token suffixes as infer.py does, without decoding the full context.
                return any(input_ids.shape[1] >= len(seq) and
                           input_ids[0, -len(seq):].tolist() == seq for seq in self.sequences)

        stop = StopOnSearch()
        stop.sequences = [tokenizer.encode(s, add_special_tokens=False) for s in
                          ['</search>', ' </search>', '</search>\n', ' </search>\n',
                           '</search>\n\n', ' </search>\n\n']]
        self.stopping = transformers.StoppingCriteriaList([stop])
        self.model = model
        print('Model loaded and ready for questions.', flush=True)

    def run(self, question, emit):
        import requests
        # Check retrieval availability for each question; reuse the startup model.
        response = requests.get(RETRIEVER_URL + '/docs', timeout=5)
        response.raise_for_status()
        question = question.strip()
        if not question.endswith('?'):
            question += '?'
        prompt = (
            'Answer the given question. You must conduct reasoning inside <think> and </think> '
            'first every time you get new information. After reasoning, if you find you lack '
            'some knowledge, you can call a search engine by <search> query </search> and it '
            'will return the top searched results between <information> and </information>. '
            'You can search as many times as your want. If you find no further external '
            'knowledge needed, you can directly provide the answer inside <answer> and '
            '</answer>, without detailed illustrations. For example, <answer> Beijing '
            '</answer>. Question: ' + question + '\n')
        if self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': prompt}], add_generation_prompt=True, tokenize=False)
        emit('prompt', text=prompt)
        trace = ''
        for turn in range(12):
            emit('status', message=f'Generating response · round {turn + 1}')
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
            context_limit = getattr(self.model.config, 'max_position_embeddings', 32768)
            if inputs.input_ids.shape[1] + 1024 > context_limit:
                raise RuntimeError('Model context limit reached. Try a shorter question.')
            with self.torch.inference_mode():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=1024, stopping_criteria=self.stopping,
                    pad_token_id=self.tokenizer.eos_token_id, do_sample=True, temperature=0.7)
            chunk = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            trace += chunk
            answers = re.findall(r'<answer>(.*?)</answer>', trace, re.DOTALL)
            queries = re.findall(r'<search>(.*?)</search>', chunk, re.DOTALL)
            emit('output', text=chunk, prefix='\n\n' if queries and not answers else '')
            if answers:
                emit('answer', text=answers[-1].strip())
                return
            if queries:
                query = queries[-1].strip()
                emit('search', query=query, round=turn + 1)
                response = requests.post(RETRIEVER_URL + '/retrieve', json={
                    'queries': [query], 'topk': 3, 'return_scores': True}, timeout=120)
                response.raise_for_status()
                docs = response.json()['result'][0]
                passages = []
                for i, item in enumerate(docs):
                    content = item['document']['contents']
                    title, _, body = content.partition('\n')
                    passages.append(f'Doc {i+1}(Title: {title}) {body}\n')
                information = ''.join(passages)
                emit('results', text=information)
                addition = f'\n\n{chunk}<information>{information}</information>\n\n'
                trace += f'<information>{information}</information>\n'
                prompt += addition
            else:
                eos = self.model.generation_config.eos_token_id
                eos_ids = eos if isinstance(eos, list) else [eos]
                if outputs[0][-1].item() in eos_ids:
                    raise RuntimeError('The model stopped without an <answer>. See model output below.')
                prompt += chunk
        raise RuntimeError('Stopped after 12 rounds without a final answer. See model output below.')


engine = Inference()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ('/', '/index.html'):
            self.send_error(404)
            return
        body = PAGE.read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != '/api/ask':
            self.send_error(404)
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 32768:
                raise ValueError('Request must contain a question (maximum 32 KB).')
            question = json.loads(self.rfile.read(length)).get('question')
            if not isinstance(question, str) or not question.strip() or len(question) > 4000:
                raise ValueError('Enter a question between 1 and 4000 characters.')
        except (ValueError, AttributeError) as exc:
            self.send_error(400, str(exc))
            return
        if not engine.lock.acquire(blocking=False):
            self.send_error(409, 'Another question is running. Try again when it finishes.')
            return
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'application/x-ndjson; charset=utf-8')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'close')
            self.end_headers()
            self.close_connection = True

            def emit(kind, **data):
                self.wfile.write((json.dumps({'type': kind, **data}) + '\n').encode())
                self.wfile.flush()

            try:
                engine.run(question, emit)
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as exc:
                self.log_error('Inference failed: %s', exc)
                emit('error', message=str(exc))
            emit('done')
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            engine.lock.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    engine.load()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print('The model and web server are ready.', flush=True)
    print(f'Search-R1 demo: http://{args.host}:{args.port}', flush=True)
    print('Press Ctrl+C to stop the web server.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
