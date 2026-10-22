# Third-party attribution

The runtime retains the existing RecBole3-style package structure and TIGER-style
T5 decoder implementation. Backbone SID maps were constructed using RQ-VAE,
LETTER and ETEGRec. The release calls the RQ-VAE-based backbone TIGER.
Refer to the corresponding papers when using their maps.

`scripts/sasrec_cf.py` contains the SASRec and PointWiseFeedForward classes used
in the experiments, extracted without algorithmic changes from the existing
sequential-model implementation. That implementation attributes its SASRec
architecture to https://github.com/pmixer/SASRec.pytorch.

Pretrained Sentence-T5 models and Amazon review data are downloaded separately;
their upstream terms apply. No pretrained weights or raw user histories are
distributed here.

No new blanket license is asserted for third-party code. The source checkout
did not provide a repository-level LICENSE file. Redistribution permissions
and applicable upstream license notices must be confirmed before public release.
