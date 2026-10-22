"""Portable orchestration of the existing PPTT, BCS-F and handoff scripts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
PAIRS=[f'{f}_{d}' for d in ['grocery','pet_supplies','video_games'] for f in ['tiger','letter','etegrec']]


def command(*args):
    subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,check=True)


def hydra_value(value):
    if isinstance(value,dict):return '{'+','.join(f'{k}:{hydra_value(v)}' for k,v in value.items())+'}'
    if isinstance(value,list):return '['+','.join(hydra_value(v) for v in value)+']'
    return json.dumps(value)


def flatten(value,prefix=''):
    for key,item in value.items():
        name=f'{prefix}.{key}' if prefix else key
        if isinstance(item,dict):yield from flatten(item,name)
        else:yield f'++{name}={hydra_value(item)}'


def settings(pair,variant,seed):
    values=json.loads((ROOT/f'configs/reproduction/{variant}/{pair}.json').read_text())
    values['runtime']=dict(seed=seed,deterministic=True,device='auto',wandb=False,
                           output_dir=f'outputs/reproduction/{variant}/{pair}_s{seed}')
    return ['model=tiger','dataset=amazon2014_retrieval',*flatten(values)]


def prepare(pair):
    from recbole3.run import compose_config
    from recbole3.config import instantiate_dataclass
    from recbole3.dataset import get_dataset_spec
    from recbole3.evaluation import EvalConfig
    cfg=compose_config(settings(pair,'PPTT',42))
    spec=get_dataset_spec('amazon2014_retrieval')
    dataset=spec.dataset_cls(instantiate_dataclass(spec.config_cls,cfg.dataset))
    dataset.prepare(eval_config=EvalConfig(protocol='full'))
    tag=pair.split('_',1)[1]
    category=cfg.dataset.category
    directory=ROOT/f'data/processed/amazon2014_retrieval/{category}/sentence'
    ids=[json.loads(x)['item_id'] for x in (directory/'items.jsonl').read_text().splitlines() if x.strip()]
    users=[json.loads(x)['user_id'] for x in (directory/'users.jsonl').read_text().splitlines() if x.strip()]
    expected=json.loads((ROOT/'artifacts/catalog_order.json').read_text())[tag]
    if ids!=expected['items'] or hashlib.sha256(json.dumps(users).encode()).hexdigest()!=expected['user_order_sha256']:
        raise ValueError('Dataset entity ordering differs from the paper SID inputs. Do not silently remap items or users.')
    out=ROOT/'outputs'/pair;out.mkdir(parents=True,exist_ok=True)
    for source in (ROOT/'artifacts/sids'/pair).glob('*.json'):
        shutil.copyfile(source,out/source.name)
    command('scripts/build_sid_text_embed.py','--token_text',out/'item_sids_words_flatoffset_pp.token_text.json',
            '--out',out/'sid_text_embed_pp.pt')


def cf_embeddings(pair,override=None):
    path=Path(override) if override else ROOT/f'artifacts/sasrec/{pair.split("_",1)[1]}/embeddings.npz'
    if not path.is_file():raise FileNotFoundError(f'Missing frozen item-SASRec embeddings: {path}')
    return path


def fit(pair,variant,seed,limit,cf_file):
    import numpy as np
    from trace_item_path_scorer import ItemPathScorer
    tag=pair.split('_',1)[1]
    sid=ROOT/'artifacts/sids'/pair/('item_sids_backbone_suffix.json' if variant=='backbone' else 'item_sids_words_flatoffset_pp_dedup.json')
    sids={int(k):tuple(v) for k,v in json.loads(sid.read_text()).items()}
    width=len(next(iter(sids.values())))
    with np.load(cf_file,allow_pickle=False) as z:
        scorer=ItemPathScorer(sids,z['items'],dict(zip(map(int,z['users']),z['profiles'])))
    root=ROOT/f'outputs/reproduction/{variant}/{pair}_s{seed}'
    ckpt=root/'checkpoints/best_model.pt'
    overrides=settings(pair,variant,seed)+['++model.num_beams=40','++trainer.batch_size=64','++trainer.mixed_precision=no']
    for split in ['valid','test']:
        command('scripts/eval_c3_prefix_handoff.py','--ckpt',ckpt,'--split',split,'--alphas','0',
                '--depths',f'{width-1},{width}','--trace','--candidate-trace',
                '--max-users',limit,'--output',root/f'{split}.json',*overrides)
        with np.load(root/f'{split}.candidates.npz') as z:data={k:z[k] for k in z.files}
        for label,depth in [('early',width-1),('full',width)]:
            offsets=[0];items=[];decoder=[];beam=[];fallback=[];users=[];targets=[];native=[]
            for row in np.flatnonzero(data['depth']==depth):
                q=int(data['query_index'][row]);lo,hi=map(int,data['offsets'][row:row+2])
                items.extend(data['items'][lo:hi]);decoder.extend(data['decoder'][lo:hi]);beam.extend(data['beam'][lo:hi])
                offsets.append(len(items));fallback.append(data['fallback'][row]);users.append(data['user_ids'][q])
                targets.append(data['targets'][q]);native.append(data['native'][q])
            items=np.asarray(items,dtype=np.int32);offsets=np.asarray(offsets,dtype=np.int64)
            cf=[]
            for start in range(0,len(users),64):
                stop=min(start+64,len(users))
                pools=[items[offsets[q]:offsets[q+1]] for q in range(start,stop)]
                cf.extend(scorer.score(users[start:stop],pools))
            destination=root/label;destination.mkdir(exist_ok=True)
            np.savez_compressed(destination/f'{split}.npz',pair=np.asarray(pair),users=users,targets=targets,
                                native=native,fallback=fallback,items=items,offsets=offsets,decoder=np.asarray(decoder,np.float32),
                                beam=beam,cf_logp=np.concatenate(cf),candidate_depth=np.asarray(depth))
    for label in ['early','full']:
        command('scripts/fit_tiger_dynamic_path_fusion.py','--valid',root/label/'valid.npz',
                '--test',root/label/'test.npz','--output',root/label/'result.json')
    from fit_tiger_dynamic_path_fusion import evaluate, METRICS
    baseline,fused=evaluate(root/'full')
    (root/'metrics.json').write_text(json.dumps(dict(
        original_checkpoint={name:value for (name,_,_),value in zip(METRICS,baseline)},
        bcsf={name:value for (name,_,_),value in zip(METRICS,fused)}),indent=2)+'\n')
    command('scripts/analyze_tiger_dynamic_path_adaptive.py','--pair-dir',root,
            '--valid-base',root/'valid.npz','--test-base',root/'test.npz','--output',root/'policy.json',
            '--max-degradation-pct','0')
    print(f'Full BCS-F: {root}/full/result.json\nAdaptive quality: {root}/policy.json')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','train','cf','evaluate','latency','summarize'])
    p.add_argument('--pair',required=True,choices=PAIRS)
    p.add_argument('--variant',choices=['backbone','PPTT'],default='PPTT')
    p.add_argument('--seed',type=int,choices=[42,123,2024],default=42)
    p.add_argument('--max-users',type=int,default=0,help='Evaluation smoke limit; 0 = all. Never use limited output as paper results.')
    p.add_argument('--warmup-users',type=int,default=32)
    p.add_argument('--cf-embeddings',type=Path,help='Override the bundled seed-42 item-SASRec embeddings')
    args=p.parse_args();pair=args.pair;tag=pair.split('_',1)[1]
    if args.stage=='prepare':prepare(pair)
    elif args.stage=='train':command('-m','recbole3.run',*settings(pair,args.variant,args.seed))
    elif args.stage=='cf':
        base='tiger_'+tag
        command('scripts/build_user_prefix_graph.py','--family','tiger','--tag',tag,
                '--sid-file',ROOT/f'artifacts/sids/{base}/item_sids_words_flatoffset_pp_dedup.json')
        command('scripts/train_trace_item_cf.py','--dataset',tag)
    elif args.stage=='evaluate':fit(pair,args.variant,args.seed,args.max_users,cf_embeddings(pair,args.cf_embeddings))
    elif args.stage=='summarize':
        import numpy as np
        rows=[]
        for seed in [42,123,2024]:
            rows.append(json.loads((ROOT/f'outputs/reproduction/{args.variant}/{pair}_s{seed}/metrics.json').read_text()))
        summary={side:{metric:dict(mean=float(np.mean([r[side][metric] for r in rows])),
                                  sample_std=float(np.std([r[side][metric] for r in rows],ddof=1)))
                       for metric in rows[0][side]} for side in rows[0]}
        print(json.dumps(summary,indent=2))
    else:
        root=ROOT/f'outputs/reproduction/{args.variant}/{pair}_s{args.seed}'
        sidname='item_sids_backbone_suffix.json' if args.variant=='backbone' else 'item_sids_words_flatoffset_pp_dedup.json'
        command('scripts/bench_trace_branch_adaptive.py','--pair',pair,'--pair-root',root,
                '--checkpoint',root/'checkpoints/best_model.pt','--sid-file',ROOT/f'artifacts/sids/{pair}/{sidname}',
                '--cf-file',cf_embeddings(pair,args.cf_embeddings),
                '--max-users',args.max_users or 512,'--warmup-users',args.warmup_users)


if __name__=='__main__':main()
