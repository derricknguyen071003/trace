"""SASRec implementation used in the experiments; see THIRD_PARTY.md."""
import numpy as np
import torch
from torch import nn

class SASRec(nn.Module):
    """Adaptation of code from
    https://github.com/pmixer/SASRec.pytorch.
    """

    def __init__(self, item_num, maxlen=128, hidden_units=64, num_blocks=1, num_heads=1, dropout_rate=0.1, initializer_range=0.02, add_head=True):
        super(SASRec, self).__init__()
        self.item_num = item_num
        self.maxlen = maxlen
        self.hidden_units = hidden_units
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.initializer_range = initializer_range
        self.add_head = add_head
        self.item_emb = nn.Embedding(item_num + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-08)
        for _ in range(num_blocks):
            new_attn_layernorm = nn.LayerNorm(hidden_units, eps=1e-08)
            self.attention_layernorms.append(new_attn_layernorm)
            new_attn_layer = nn.MultiheadAttention(hidden_units, num_heads, dropout_rate)
            self.attention_layers.append(new_attn_layer)
            new_fwd_layernorm = nn.LayerNorm(hidden_units, eps=1e-08)
            self.forward_layernorms.append(new_fwd_layernorm)
            new_fwd_layer = PointWiseFeedForward(hidden_units, dropout_rate)
            self.forward_layers.append(new_fwd_layer)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights.

        Examples:
        https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/models/gpt2/modeling_gpt2.py#L454
        https://recbole.io/docs/_modules/recbole/model/sequential_recommender/sasrec.html#SASRec
        """
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask):
        seqs = self.item_emb(input_ids)
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(input_ids.shape[1])), [input_ids.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(seqs.device))
        seqs = self.emb_dropout(seqs)
        timeline_mask = torch.Tensor(input_ids == 0)
        seqs *= ~timeline_mask.unsqueeze(-1)
        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool).to(seqs.device))
        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            (mha_outputs, _) = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)
        outputs = self.last_layernorm(seqs)
        if self.add_head:
            outputs = torch.matmul(outputs, self.item_emb.weight.transpose(0, 1))
        return outputs

class PointWiseFeedForward(nn.Module):

    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs

from pathlib import Path
from sasrec_prefix_contract import right_pad_sequences

def maps(graph: Path):
    users = {}
    with (graph / 'user_id_map.tsv').open() as handle:
        next(handle)
        for line in handle:
            (row, user) = line.rstrip().split('\t')
            users[int(user)] = int(row)
    prefixes = {}
    with (graph / 'prefix_id_map.tsv').open() as handle:
        next(handle)
        for line in handle:
            (row, _, prefix) = line.rstrip().split('\t')
            prefixes[tuple(json.loads(prefix))] = int(row)
    return (users, prefixes)

def sequences(path: Path):
    result = {}
    with path.open() as handle:
        for line in handle:
            values = [int(value) for value in line.split()]
            result[values[0]] = [value + 1 for value in values[1:]]
    return result

@torch.inference_mode()
def hidden_gpu(model, rows, max_length, device, chunk_size):
    outputs = []
    for start in range(0, len(rows), chunk_size):
        part = [row[-max_length:] for row in rows[start:start + chunk_size]]
        (padded, last) = right_pad_sequences(part)
        inputs = torch.as_tensor(padded, dtype=torch.long, device=device)
        encoded = model(inputs, (inputs != 0).float())
        index = torch.as_tensor(last, dtype=torch.long, device=device)
        outputs.append(encoded[torch.arange(len(part), device=device), index].float())
    return torch.cat(outputs)
