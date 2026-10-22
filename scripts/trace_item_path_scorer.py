"""Convert frozen item-SASRec logits to conditional SID branch probabilities."""
import numpy as np

class ItemPathScorer:
    def __init__(self,sids,vectors,profiles):
        self.vectors=vectors;self.profiles=profiles
        assert set(sids)==set(range(len(vectors)))
        codes=np.asarray([sids[i] for i in range(len(vectors))])
        assert len(np.unique(codes,axis=0))==len(vectors)
        self.groups=[]
        for d in range(1,codes.shape[1]+1):
            _,inverse=np.unique(codes[:,:d],axis=0,return_inverse=True)
            order=np.argsort(inverse,kind='stable')
            sorted_ids=inverse[order]
            starts=np.r_[0,np.flatnonzero(np.diff(sorted_ids))+1]
            self.groups.append((inverse,order,sorted_ids,starts))

    def score(self,users,pools):
        states=np.stack([self.profiles[int(u)] for u in users])
        logits=states@self.vectors.T
        root_max=logits.max(1)
        root=root_max+np.log(np.exp(logits-root_max[:,None]).sum(1))
        previous=[np.full(len(pool),root[q]) for q,pool in enumerate(pools)]
        result=[np.empty((len(pool),len(self.groups)),np.float32) for pool in pools]
        for d,(inverse,order,sorted_ids,starts) in enumerate(self.groups):
            sorted_logits=logits[:,order]
            maxima=np.maximum.reduceat(sorted_logits,starts,axis=1)
            mass=maxima+np.log(np.add.reduceat(np.exp(sorted_logits-maxima[:,sorted_ids]),starts,axis=1))
            for q,pool in enumerate(pools):
                current=mass[q,inverse[pool]]
                result[q][:,d]=current-previous[q]
                previous[q]=current
        # Chain rule: complete-path log probability equals item softmax log prob.
        for q,pool in enumerate(pools):
            if len(pool):
                assert np.allclose(result[q].sum(1),logits[q,pool]-root[q],atol=2e-5,rtol=2e-5)
        return result
