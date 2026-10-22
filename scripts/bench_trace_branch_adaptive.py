"""Live, cache-preserving branch item-CF handoff; end-to-end batch-one timing."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from collections import defaultdict
from recbole3.config import instantiate_dataclass
from recbole3.dataset import get_dataset_spec
from recbole3.model import get_model_spec
from recbole3.run import compose_config
from recbole3.utils import require_component_cfg, require_component_name
from analyze_tiger_dynamic_path_adaptive import apply_rule, rankings_and_features, contributions, metrics, load_npz
from eval_c3_prefix_handoff import _excluded, _records_list, _fill
from recbole3.dataset import USER_ID, ITEM_ID
from trace_item_path_scorer import ItemPathScorer

class SelectiveDecoder:

    def __init__(self, model, emb_dir):
        self.model = model
        self.t5 = model._t5_module()
        self.codec = model._sid_codec()
        self.L = self.codec.n_digit
        self.device = next(self.t5.parameters()).device
        self.beams = model.config.num_beams
        self.buckets = defaultdict(list)
        for (item, tokens) in self.codec.item_to_tokens.items():
            for d in range(1, self.L + 1):
                self.buckets[tokens[:d]].append(item)

def setup(overrides, ckpt, emb_dir):
    cfg = compose_config(overrides=overrides)
    (dc, mc, tc) = [require_component_cfg(cfg, key) for key in ('dataset', 'model', 'trainer')]
    ds = get_dataset_spec(require_component_name(dc, 'dataset'))
    ms = get_model_spec(require_component_name(mc, 'model'))
    dataset = ds.dataset_cls(instantiate_dataclass(ds.config_cls, dc))
    model = ms.model_cls(instantiate_dataclass(ms.config_cls, mc))
    trainer = ms.trainer_cls(instantiate_dataclass(ms.trainer_config_cls, tc))
    prepared = ms.model_data_cls.from_task_dataset(dataset.prepare(eval_config=trainer.config.eval), model_config=mc)
    model.build_eval_collator(prepared)
    model.load_state_dict(torch.load(ckpt, weights_only=True, map_location='cpu'), strict=True)
    model.cuda().eval()
    return (SelectiveDecoder(model, emb_dir), trainer, prepared)
def generate(decoder, inputs, *, stop=None, output_scores=False):
    model = decoder.model
    user_ids = inputs.get('user_ids')
    kwargs = model._encoder_kwargs(inputs['input_ids'], inputs['attention_mask'], user_ids, inputs.get('literal_input_ids'), inputs.get('literal_attention_mask'))
    generation = {'max_new_tokens': decoder.L + 1, 'num_beams': decoder.beams, 'num_return_sequences': decoder.beams, 'return_dict_in_generate': True, 'output_scores': output_scores, 'length_penalty': 0.0, 'use_cache': True}
    if stop is not None:
        generation['stopping_criteria'] = StoppingCriteriaList([stop])
    return decoder.t5.generate(**generation, **kwargs)

def full_ranking(decoder, generated, excluded):
    sequences = generated.sequences.reshape(1, decoder.beams, -1)
    (ranking, used) = ([], set())
    for beam in range(decoder.beams):
        tokens = sequences[0, beam, 1:1 + decoder.L].detach().cpu().tolist()
        item = decoder.codec.token_tuple_to_item(tokens)
        if item is None or item in excluded or item in used:
            continue
        ranking.append(item)
        used.add(item)
        if len(ranking) == 10:
            break
    _fill(ranking, excluded, decoder.codec.fallback_item_ids)
    return ranking

def check_pool_parity(data, reference, depth, atol=1e-5):
    """Allow only float32 near-ties at the beam cutoff, never substitute trace items.

    Cached shallow generation uses length penalty one, while live ancestry
    contains cumulative log-probabilities. Compare in the cached score units.
    """
    ref=reference['data'];lo,hi=reference['span']
    order=np.argsort(ref['items'][lo:hi])
    old_items=ref['items'][lo:hi][order]
    old_scores=ref['decoder'][lo:hi][order]
    live_items=data['items'];live_scores=data['decoder']/depth
    shared=np.intersect1d(live_items,old_items)
    if not len(shared):raise ValueError('Live and trace pools have no shared items')
    live_shared=live_scores[np.searchsorted(live_items,shared)]
    old_shared=old_scores[np.searchsorted(old_items,shared)]
    if not np.allclose(live_shared,old_shared,atol=atol,rtol=0):
        raise ValueError('Live prefix scores differ substantively from saved trace')
    if np.array_equal(live_items,old_items):return None
    live_only=np.setdiff1d(live_items,old_items)
    trace_only=np.setdiff1d(old_items,live_items)
    cutoff_live=float(live_scores.min());cutoff_trace=float(old_scores.min())
    if abs(cutoff_live-cutoff_trace)>atol:
        raise ValueError('Live and trace beam cutoffs differ substantively')
    for items,scores,changed,cutoff in [(live_items,live_scores,live_only,cutoff_live),(old_items,old_scores,trace_only,cutoff_trace)]:
        if len(changed) and np.any(np.abs(scores[np.searchsorted(items,changed)]-cutoff)>atol):
            raise ValueError('Live pool difference is not a numerical tie at the beam cutoff')
    return dict(live_only=live_only.tolist(),trace_only=trace_only.tolist(),
                cutoff_score_difference=abs(cutoff_live-cutoff_trace),absolute_tolerance=atol)

def pool_data(decoder, record, prefixes, scores, cf):
    excluded = _excluded(record)
    by_item = {}
    for beam,(prefix,score) in enumerate(zip(prefixes,scores)):
        for item in decoder.buckets.get(tuple(prefix),()):
            if item not in excluded and (item not in by_item or score > by_item[item][0]):
                by_item[item]=(float(score),beam)
    items=np.asarray(sorted(by_item),np.int32)
    if not len(items): raise ValueError('No valid candidate items in live beam')
    fallback=[i for i in decoder.codec.fallback_item_ids if i not in excluded][:10]
    if len(fallback)!=10: raise ValueError('Insufficient fallback items')
    return dict(items=items,decoder=np.asarray([by_item[i][0] for i in items]),
                beam=np.asarray([by_item[i][1] for i in items]),cf_logp=cf[items],
                offsets=np.asarray([0,len(items)]),fallback=np.asarray([fallback]),
                native=np.asarray([fallback]),targets=np.asarray([record[ITEM_ID]]))

class LiveStop(StoppingCriteria):
    def __init__(self,decoder,record,scorer,alpha,policy,reference):
        self.decoder=decoder;self.record=record;self.scorer=scorer;self.alpha=alpha
        self.policy=policy;self.reference=reference
        self.previous=[()] * decoder.beams
        self.raw=np.r_[0.,np.full(decoder.beams-1,-1e9)]
        self.cf=None;self.pred=None;self.stopped=False;self.parity=False;self.roundoff_case=None

    def __call__(self,input_ids,scores,**kwargs):
        depth=input_ids.shape[1]-1
        if depth>self.decoder.L-1:
            return torch.zeros(len(input_ids),dtype=torch.bool,device=input_ids.device)
        prefixes=[tuple(r) for r in input_ids[:self.decoder.beams,1:].cpu().tolist()]
        parent_rows={p:i for i,p in reversed(list(enumerate(self.previous)))}
        parents=[parent_rows[p[:-1]] for p in prefixes]
        emitted=[p[-1] for p in prefixes]
        values=scores[-1][torch.tensor(parents,device=input_ids.device),torch.tensor(emitted,device=input_ids.device)].float().cpu().numpy()
        raw=self.raw[parents]+values
        self.previous=prefixes;self.raw=raw
        if depth==self.decoder.L-1:
            user=int(self.record[USER_ID])
            self.cf=self.scorer.score([user],[np.arange(len(self.scorer.vectors))])[0]
            data=pool_data(self.decoder,self.record,prefixes,raw,self.cf)
            pred,_,features=rankings_and_features(data,self.alpha)
            current=scores[-1].float();probs=current.softmax(-1)
            entropy=(-(probs*probs.clamp_min(1e-30).log()).sum(-1)/torch.isfinite(current).sum(-1).clamp_min(2).float().log()).mean().item()
            features=np.column_stack((features,[entropy]))
            # Prevent a scheduled trace replay from masquerading as a live controller.
            # Live candidate pools must agree with the frozen quality experiment.
            ref=self.reference;lo,hi=ref['span']
            try:
                self.roundoff_case=check_pool_parity(data,ref,depth)
            except ValueError:
                debug=Path('outputs/analysis/trace_branch_adaptive')/str(ref['data']['pair'])/'parity_failure.npz'
                np.savez_compressed(debug,user=user,prefixes=np.asarray(prefixes),raw=raw,
                                    live_items=data['items'],live_decoder=data['decoder'],
                                    reference_items=ref['data']['items'][lo:hi],
                                    reference_decoder=ref['data']['decoder'][lo:hi],
                                    reference_beam=ref['data']['beam'][lo:hi],
                                    eos_token_id=self.decoder.t5.generation_config.eos_token_id)
                raise ValueError(f'Live early pool/score parity failed: user={user}; diagnostic={debug}')
            self.parity=True
            self.stopped=bool(apply_rule(features,self.policy['gate'])[0])
            self.pred=pred[0]
        return torch.full((len(input_ids),),self.stopped,dtype=torch.bool,device=input_ids.device)

def main():
    p=argparse.ArgumentParser();p.add_argument('--pair',required=True);p.add_argument('--max-users',type=int,default=512);p.add_argument('--warmup-users',type=int,default=32);p.add_argument('--smoke',action='store_true');p.add_argument('--checkpoint');p.add_argument('--pair-root',type=Path);p.add_argument('--sid-file',type=Path);p.add_argument('--cf-file',type=Path)
    args=p.parse_args();pair=args.pair;root=args.pair_root or Path('outputs/analysis/trace_branch_adaptive')/pair
    policy=json.loads((root/'policy.json').read_text())
    alpha=lambda label:np.asarray(json.loads((root/label/'result.json').read_text())['learned_alpha_by_depth'])
    early_alpha,full_alpha=alpha('early'),alpha('full')
    early=load_npz(root/'early/test.npz');user_rows={int(u):q for q,u in enumerate(early['users'])}
    sids={int(i):tuple(s) for i,s in json.loads((args.sid_file or Path('outputs')/pair/'item_sids_words_flatoffset_pp_dedup.json').read_text()).items()}
    with np.load(args.cf_file or Path('outputs/analysis/trace_item_cf')/pair.split('_',1)[1]/'embeddings.npz') as z:
        scorer=ItemPathScorer(sids,z['items'],dict(zip(map(int,z['users']),z['profiles'])))
    category={'grocery':'Grocery_and_Gourmet_Food','pet_supplies':'Pet_Supplies','video_games':'Video_Games'}[pair.split('_',1)[1]]
    decoder,trainer,prepared=setup(['model=tiger','model.num_beams=40','dataset=amazon2014_retrieval',f'dataset.category={category}',f"model.sid_file={args.sid_file or Path('outputs')/pair/'item_sids_words_flatoffset_pp_dedup.json'}",'trainer.batch_size=1','trainer.mixed_precision=no','runtime.device=auto'],args.checkpoint or f'outputs/c1text_pp_PPTT_matched/{pair}/checkpoints/best_model.pt','')
    method=trainer.create_evaluation_method(prepared)
    loader=trainer.build_dataloader(prepared.get_eval_dataset('test'),method.build_eval_collate_fn(decoder.model,prepared),shuffle=False)
    timings={m:[] for m in ['decoder_only','full_branch_cf','adaptive_branch_cf']}
    predictions={m:[] for m in timings};targets=[];users=[];depths=[];roundoff_cases=[]
    with torch.inference_mode():
        for index,(inputs,records) in enumerate(loader):
            if index>=args.max_users:break
            record=_records_list(records)[0];user=int(record[USER_ID]);q=user_rows[user]
            reference=dict(data=early,span=tuple(map(int,early['offsets'][q:q+2])))
            inputs={k:v.cuda() for k,v in inputs.items()}
            modes=list(timings);modes=modes[index%3:]+modes[:index%3]
            for mode in modes:
                torch.cuda.synchronize();started=time.perf_counter()
                if mode=='decoder_only':
                    generated=generate(decoder,inputs,output_scores=False)
                    pred=full_ranking(decoder,generated,_excluded(record))
                else:
                    stop=LiveStop(decoder,record,scorer,early_alpha,policy,reference) if mode=='adaptive_branch_cf' else None
                    generated=generate(decoder,inputs,stop=stop,output_scores=True)
                    if stop is not None and stop.stopped:
                        pred=stop.pred;depths.append(decoder.L-1)
                    else:
                        prefixes=generated.sequences[:,1:decoder.L+1].cpu().tolist()
                        raw=generated.sequences_scores.float().cpu().numpy()
                        cf=stop.cf if stop is not None and stop.cf is not None else scorer.score([user],[np.arange(len(scorer.vectors))])[0]
                        data=pool_data(decoder,record,prefixes,raw,cf)
                        pred=rankings_and_features(data,full_alpha)[0][0]
                        if stop is not None:depths.append(decoder.L)
                    if stop is not None and not stop.parity:raise ValueError('Live gate not evaluated')
                    if stop is not None and stop.roundoff_case is not None:
                        roundoff_cases.append(dict(user=user,**stop.roundoff_case))
                torch.cuda.synchronize();elapsed=time.perf_counter()-started
                if index>=args.warmup_users:timings[mode].append(elapsed)
                predictions[mode].append(pred)
            targets.append(int(record[ITEM_ID]))
            users.append(user)
    base=contributions(np.asarray(predictions['decoder_only']),np.asarray(targets))
    result=dict(pair=pair,users=len(targets),gpu=torch.cuda.get_device_name(),batch_size=1,beam=40,decoder_seed=42,cf_training_seed=42,
                timing_scope='live encoder + beam generation + CF branch masses + descendant lookup + validation-frozen gate + ranking; continued users reuse same generate KV cache; model loading excluded',
                depth_counts={str(d):depths.count(d) for d in [decoder.L-1,decoder.L]},nominal_step_reduction_pct=100*depths.count(decoder.L-1)/len(depths)/decoder.L,methods={},
                parity_audit=dict(exact_pool_queries=len(targets)-len(roundoff_cases),roundoff_boundary_queries=len(roundoff_cases),cases=roundoff_cases,policy='Keep actual live candidates; permit only cutoff-score near-ties within 1e-5, with shared-score parity; reject all substantive drift'))
    for mode,values in timings.items():
        result['methods'][mode]=dict(mean_ms=1000*float(np.mean(values)),median_ms=1000*float(np.median(values)),p95_ms=1000*float(np.quantile(values,.95)),timed_users=len(values),**metrics(contributions(np.asarray(predictions[mode]),np.asarray(targets)),base))
    adaptive=result['methods']['adaptive_branch_cf']['mean_ms']
    result['latency_saving_vs_decoder_pct']=100*(1-adaptive/result['methods']['decoder_only']['mean_ms'])
    result['latency_saving_vs_full_branch_cf_pct']=100*(1-adaptive/result['methods']['full_branch_cf']['mean_ms'])
    stem='smoke' if args.smoke else 'live'
    (root/f'{stem}_latency.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(root/f'{stem}_predictions.npz',users=users,targets=targets,exit_depth=depths,**{m:np.asarray(v) for m,v in predictions.items()},**{m+'_seconds':v for m,v in timings.items()})
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
