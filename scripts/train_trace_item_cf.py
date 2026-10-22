"""Ordinary item-SASRec control; same purchases as frozen prefix SASRec."""
import argparse,json,sys,random
from pathlib import Path
import numpy as np
import torch
from sasrec_cf import maps, sequences, SASRec, hidden_gpu
from sasrec_prefix_contract import right_pad_sequences
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from recbole3.checkpoint_storage import save_checkpoint

def training_batch(rows,device):
    padded,_=right_pad_sequences([s[:-1] for s in rows])
    x=np.asarray(padded,dtype=np.int64)
    labels=np.full(x.shape,-100,dtype=np.int64)
    for j,s in enumerate(rows):labels[j,:len(s)-1]=s[1:]
    return torch.as_tensor(x,device=device),torch.as_tensor(labels,device=device)

def train(seq,item_count,params,device='cuda',epochs=12):
    model=SASRec(item_count,maxlen=params['max_length'],hidden_units=64,num_blocks=2,num_heads=1,dropout_rate=.2,add_head=False).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    histories=[s for s in seq.values() if len(s)>1]
    for epoch in range(epochs):
        model.train();total=0.
        for start in range(0,len(histories),64):
            if start==0:random.shuffle(histories)
            rows=histories[start:start+64]
            x,y=training_batch(rows,device)
            h=model(x,(x!=0).float())
            # Same full-softmax CE as SeqRec, excluding padding before projection.
            valid=y!=-100
            loss=torch.nn.functional.cross_entropy(h[valid]@model.item_emb.weight.T,y[valid])
            opt.zero_grad();loss.backward();opt.step();total+=float(loss.detach())
        print(dict(epoch=epoch+1,loss_sum=total),flush=True)
    return model.eval()

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);a=p.parse_args()
    graph=Path('data/user_prefix_graph')/('tiger_'+a.dataset)
    users,prefixes=maps(graph);history=sequences(graph/'sasrec_train.txt')
    sidpath=Path('outputs')/('tiger_'+a.dataset)/'item_sids_words_flatoffset_pp_dedup.json'
    sids={int(i):tuple(s) for i,s in json.loads(sidpath.read_text()).items()}
    assert len(set(sids.values()))==len(sids),'Item control requires unique full SIDs'
    leaves={prefixes[s]+1:i+1 for i,s in sids.items() if s in prefixes}
    seq={u:[leaves[e] for e in events if e in leaves] for u,events in history.items()}
    assert all(seq.values())
    params=dict(max_length=max(map(len,seq.values())),num_blocks=2,dropout_rate=.2,lr=.001,batch_size=64)
    out=Path('outputs/analysis/trace_item_cf')/a.dataset
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite existing artifacts: {out}')
    out.mkdir(parents=True,exist_ok=True)
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    model=train(seq,len(sids),params)
    embeddings=hidden_gpu(model,[seq[users[u]] for u in users],params['max_length'],'cuda',128).cpu().numpy()
    np.savez_compressed(out/'embeddings.npz',users=np.asarray(list(users)),profiles=embeddings,items=model.item_emb.weight.detach().cpu().numpy()[1:])
    save_checkpoint({'state_dict':model.state_dict(),'params':params},out/'sasrec_model.pt')
    (out/'protocol.json').write_text(json.dumps(dict(dataset=a.dataset,seed=42,epochs=12,params=params,source=str(graph),history='Full-leaf events only from identical outer-training purchase histories; no validation/test events'),indent=2))
if __name__=='__main__':main()
