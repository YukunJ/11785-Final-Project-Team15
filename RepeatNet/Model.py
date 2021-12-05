import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
import pickle
from Common.BilinearAttention import *

base_data_path = './datasets/demo/'
with open(base_data_path+'digi_item_to_side_index.pkl', 'rb') as f_handle:
    INDEX_LIST = [0] + list(pickle.load(f_handle))

def gru_forward(gru, input, lengths, state=None, batch_first=True):
    gru.flatten_parameters()
    input_lengths, perm = torch.sort(lengths, descending=True)

    input = input[perm]
    if state is not None:
        state = state[perm].transpose(0, 1).contiguous()

    total_length=input.size(1)
    if not batch_first:
        input = input.transpose(0, 1)  # B x L x N -> L x B x N

    packed = torch.nn.utils.rnn.pack_padded_sequence(input, input_lengths.cpu(), batch_first)

    outputs, state = gru(packed, state)
    outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=batch_first, total_length=total_length)  # unpack (back to padded)

    _, perm = torch.sort(perm, descending=False)
    if not batch_first:
        outputs = outputs.transpose(0, 1)
    outputs=outputs[perm]
    state = state.transpose(0, 1)[perm]

    return outputs, state

def build_map(b_map, max=None):
    batch_size, b_len = b_map.size()
    if max is None:
        max=b_map.max() + 1
    if torch.cuda.is_available():
        b_map_ = torch.cuda.FloatTensor(batch_size, b_len, max).fill_(0)
    else:
        b_map_ = torch.zeros(batch_size, b_len, max)
    b_map_.scatter_(2, b_map.unsqueeze(2), 1.)
    # b_map_[:, :, 0] = 0.
    b_map_.requires_grad=False
    return b_map_

class RepeatNet(nn.Module):
    def __init__(self, embedding_size, hidden_size, item_vocab_size):
        """
        添加side_size 作为process side information的embedding
        """
        super(RepeatNet, self).__init__()

        self.embedding_size=embedding_size
        self.hidden_size=hidden_size
        self.item_vocab_size=item_vocab_size
        """魔改测试, 去掉side"""
        #self.side_size=side_size

        self.explore_feature_raw_trans = nn.Linear(2*hidden_size, hidden_size)
        self.item_emb = nn.Embedding(item_vocab_size, embedding_size, padding_idx=0)
        """魔改测试, 去掉side"""
        #self.size_emb = nn.Embedding(side_size, embedding_size, padding_idx=0)
        self.enc = nn.GRU(embedding_size, int(hidden_size), num_layers=1, bidirectional=False, batch_first=True)
        self.side_enc = nn.GRU(embedding_size, int(hidden_size), num_layers=1, bidirectional=False, batch_first=True)
        """魔改测试, 去掉side, 从2倍hidden调到1倍"""
        self.mode_attn = BilinearAttention(hidden_size, hidden_size, hidden_size)
        self.mode = nn.Linear(hidden_size, 2)

        self.repeat_attn = BilinearAttention(hidden_size, hidden_size, hidden_size)
        self.explore_attn = BilinearAttention(hidden_size, hidden_size, hidden_size)
        self.explore = nn.Linear(hidden_size, item_vocab_size)

    def model(self, data):
        batch_size = data['item_seq'].size(0)
        mask = data['item_seq'].ne(0)
        lengths = mask.float().sum(dim=-1).long()
        """魔改测试, 去掉side"""
        item_seq_embs = F.dropout(self.item_emb(data['item_seq']), p=0.5, training=self.training)
        #side_seq_embs = F.dropout(self.item_emb(data['side_seq']), p=0.5, training=self.training)

        output_item, state_item = gru_forward(self.enc, item_seq_embs, lengths, batch_first=True)
        #output_side, state_side = gru_forward(self.side_enc, side_seq_embs, lengths, batch_first=True)
        
        #state_item = torch.cat([state_item[:, 0, :], state_item[:, 1, :]], dim=1)
        #state_side = torch.cat([state_side[:, 0, :], state_side[:, 1, :]], dim=1)
        #output, state= torch.cat([output_item, output_side], dim=2), torch.cat([state_item[:, 0, :], state_side[:, 0, :]], dim=1)
        """魔改测试, 去掉side"""
        output, state = output_item, state_item
        state = F.dropout(state, p=0.5, training=self.training)
        output = F.dropout(output, p=0.5, training=self.training)

        explore_feature, attn, norm_attn = self.explore_attn(state.reshape(batch_size, -1).unsqueeze(1), output, output, mask=mask.unsqueeze(1))
        explore_feature_transformed = self.explore_feature_raw_trans(torch.cat([state.squeeze(1), explore_feature.squeeze(1)], dim=1))
        b = self.item_emb.weight
        """魔改测试, 去掉side"""
        #cat_b = self.size_emb.weight[INDEX_LIST]
        #b_combined = torch.cat((b, cat_b), dim=1)
        #p_explore = self.explore(explore_feature.squeeze(1))
        #p_explore = torch.matmul(explore_feature_transformed, b_combined.transpose(1, 0))
        p_explore = torch.matmul(explore_feature_transformed, b.transpose(1, 0))
        explore_mask=torch.bmm((data['item_seq']>0).float().unsqueeze(1), data['source_map']).squeeze(1)
        p_explore = p_explore.masked_fill(explore_mask.bool(), float('-inf')) # not sure we need to mask this out, depends on experiment results
        p_explore = F.softmax(p_explore, dim=-1)

        _, p_repeat = self.repeat_attn.score(state.reshape(batch_size, -1).unsqueeze(1), output, mask=mask.unsqueeze(1))
        p_repeat=torch.bmm(p_repeat, data['source_map']).squeeze(1)

        mode_feature, attn, norm_attn = self.mode_attn(state.reshape(batch_size, -1).unsqueeze(1), output, output, mask=mask.unsqueeze(1))
        p_mode=F.softmax(self.mode(mode_feature.squeeze(1)), dim=-1)

        p = p_mode[:, 0].unsqueeze(-1)*p_explore + p_mode[:, 1].unsqueeze(-1)*p_repeat

        return p

    def do_train(self, data):
        scores=self.model(data)
        loss = F.nll_loss((scores+1e-8).log(), data['item_tgt'].reshape(-1), ignore_index=0) #0 is used as padding
        return loss

    def do_infer(self, data):
        scores = self.model(data).cpu()
        scores, index=torch.sort(scores, dim=-1, descending=True)
        return scores, index

    def forward(self, data, method='mle_train'):
        data['source_map'] = build_map(data['item_seq'], max=self.item_vocab_size)
        if method == 'train':
            return self.do_train(data)
        elif method == 'infer':
            return self.do_infer(data)
