import json,sys,glob,statistics
def stats(files, label):
    ms_step=[]; tok_step=[]; ms_tok=[]
    for f in files:
        for l in open(f,encoding='utf-8'):
            r=json.loads(l)
            if r.get('kind')!='run' or r['id']=='long16k': continue
            t=r['timings']; c=r.get('client') or {}
            n=t.get('predicted_n') or c.get('n_gen'); acc=t.get('draft_n_accepted') or 0
            dec = c.get('decode_s') or t['predicted_ms']/1000
            steps=n-acc
            ms_step.append(1000*dec/steps); tok_step.append(n/steps); ms_tok.append(1000*dec/n)
    print(f"{label:45s} ms/step {statistics.median(ms_step):5.1f}  tokens/step {statistics.median(tok_step):4.2f}  ms/token {statistics.median(ms_tok):5.1f}")
L='bench/logs/'
stats(sorted(glob.glob(L+'*_strata_qwen38.jsonl'))[-1:], 'Strata default (tail 7, MTP)')
stats(sorted(glob.glob(L+'*_strata_qwen38_tail0.jsonl'))[-1:], 'Strata tail 0, MTP')
stats(sorted(glob.glob(L+'*_strata_qwen38_tail0_draftoff.jsonl'))[-1:], 'Strata tail 0, no MTP')
for tag,lab in [('ours__aic_build_','ours 0010 aic (MTP2)'),('ours_before_0010__Q4_head','ours before 0010 (MTP2)'),('ours_0010__no_draft','ours 0010 no draft'),('ours_before_0010__no_draft','ours before 0010 no draft'),('llama_cpp_best','llama.cpp best (MTP3)'),('llama_cpp_no_draft','llama.cpp no draft')]:
    fs=[f for f in glob.glob(L+'20261108-1[23]*_ms_Qwen3.8-Flas_'+tag+'*.jsonl')]
    stats(sorted(fs)[-3:], lab)
