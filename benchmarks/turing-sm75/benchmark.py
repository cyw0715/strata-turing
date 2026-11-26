import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('label')
parser.add_argument('--source', type=Path, required=True, help='Plain UTF-8 source text for the prompt')
parser.add_argument('--prompt', type=int, default=8192)
parser.add_argument('--output', type=int, default=256)
parser.add_argument('--skip-arithmetic', action='store_true')
parser.add_argument('--config', type=Path, required=True)
parser.add_argument('--port', type=int, default=8080)
parser.add_argument('--prompt-id', help='Keep this fixed across fresh-engine comparison runs')
parser.add_argument('--save-text', type=Path)
parser.add_argument('--trace-marker', type=Path)
parser.add_argument('--trace-prefill', action='store_true')
args = parser.parse_args()
cfg = json.loads(args.config.read_text())
token_dir = Path(cfg['tokenizer'])
from serve.frontend import ChatTemplate
from strata_tokenizer import Tokenizer
vocab = json.loads((token_dir/'vocab.json').read_text())
tokens = [''] * len(vocab)
for token, index in vocab.items():
    tokens[index] = token
meta = json.loads((token_dir/'tokenizer.json').read_text())
tok = Tokenizer(tokens, (token_dir/'merges.txt').read_text().split('\n'),
                json.loads((token_dir/'token_type.json').read_text()), meta['pre'], meta['special_ids'])
template = ChatTemplate(token_dir/'chat_template.jinja')
source = args.source.read_text(encoding='utf-8')

def count(text):
    rendered = template.render([{'role':'user','content':text}], enable_thinking=False, reasoning_effort='none')
    return len(tok.encode(rendered, parse_special=True))

def call(name, prompt, limit):
    payload = {'model':cfg['model_name'], 'messages':[{'role':'user','content':prompt}],
               'reasoning_effort':'none', 'chat_template_kwargs':{'enable_thinking':False,'reasoning_effort':'none'},
               'temperature':0.0 if name=='arithmetic' else 0.4, 'seed':42, 'max_tokens':limit,
               'stream':True, 'stream_options':{'include_usage':True}}
    headers = {'Content-Type':'application/json'}
    if cfg.get('api_key'):
        headers['Authorization'] = 'Bearer '+cfg['api_key']
    request = urllib.request.Request(f'http://127.0.0.1:{args.port}/v1/chat/completions',
                                    data=json.dumps(payload).encode(), headers=headers)
    start = time.perf_counter()
    text, timing, usage, finish, first = '', {}, {}, None, None
    if args.trace_marker and args.trace_prefill and name.startswith('bench'):
        args.trace_marker.touch()
    with urllib.request.urlopen(request, timeout=300) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            data = line[6:].strip()
            if data == b'[DONE]':
                break
            chunk = json.loads(data)
            if chunk.get('error'):
                raise RuntimeError(chunk['error'])
            timing = chunk.get('timings') or timing
            usage = chunk.get('usage') or usage
            for choice in chunk.get('choices', []):
                content = choice.get('delta',{}).get('content') or ''
                if content and first is None:
                    first = time.perf_counter()-start
                    if args.trace_marker and name.startswith('bench'):
                        if args.trace_prefill: args.trace_marker.unlink(missing_ok=True)
                        else: args.trace_marker.touch()
                text += content
                finish = choice.get('finish_reason') or finish
    result = {'label':args.label,'case':name,'wall_s':round(time.perf_counter()-start,3),
              'first_visible_s':first,'usage':usage,'timings':timing,'finish':finish,
              'visible_chars':len(text),'sha256':hashlib.sha256(text.encode()).hexdigest()}
    if args.trace_marker and name.startswith('bench'):
        args.trace_marker.unlink(missing_ok=True)
    if args.save_text and name.startswith('bench'):
        args.save_text.write_text(text, encoding='utf-8')
    if name == 'arithmetic':
        result['answer'] = text
        result['passed'] = text.strip() == '408'
        if not result['passed']:
            raise RuntimeError(result)
    if name.startswith('bench'):
        assert usage['prompt_tokens'] == args.prompt, result
        assert usage.get('prompt_tokens_details',{}).get('cached_tokens',0) == 0, result
        assert usage['completion_tokens'] == limit and len(text)>100, result
    print(json.dumps(result,ensure_ascii=False),flush=True)

if not args.skip_arithmetic:
    call('arithmetic','Compute 17 times 24. Return only the integer answer.',32)
intro = ('Run '+(args.prompt_id or args.label)+'. Produce 100 numbered engineering review entries, each at least 100 words, '
         'on the code below. Include concrete examples and test ideas. Start immediately with entry 1 '
         'and continue until all entries are written.\n\n')
lo,hi=0,len(source)
while lo<hi:
    mid=(lo+hi+1)//2
    if count(intro+source[:mid])<=args.prompt:
        lo=mid
    else:
        hi=mid-1
prompt=intro+source[:lo]
for _ in range(32):
    n=count(prompt)
    if n==args.prompt:
        break
    assert n<args.prompt,n
    prompt+=' X'
assert count(prompt)==args.prompt
call('bench'+str(args.prompt),prompt,args.output)
