import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import ipdb

class MatrixAttention(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out)

    def forward(self, query, keys):
        # query: [h_t || c_t]; (bsz, max_abstract_len, d_hidden*3)
        # keys: embedded x
        keys, key_lens = keys # (bsz, max entity num, d_hidden), (bsz,)
        bsz, max_n_elem, _ = keys.size()
        # TODO
        mask = torch.arange(0, max_n_elem).unsqueeze(0).repeat(bsz, 1).long().cuda()
        mask = (mask >= key_lens.unsqueeze(1)).unsqueeze(1) # (bsz, max entity num)

        # Project query (d_hidden*3 or d_hidden*2) to fixed size (d_hidden)
        query = self.proj(query) # (bsz, max abstract len, d_hidden)

        attn = torch.bmm(query, keys.transpose(1, 2)) # (bsz, max abstract len, max entity num)
        attn.masked_fill_(mask, -float('inf'))
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, keys)
        return out, attn


class BahdanauAttention(nn.Module):
    def __init__(self, num_units, query_size, memory_size):
        super(BahdanauAttention, self).__init__()

        self._num_units = num_units
        self._softmax = nn.Softmax()

        self.query_layer = nn.Linear(query_size, num_units, bias=False)
        self.memory_layer = nn.Linear(memory_size, num_units, bias=False)
        self.alignment_layer = nn.Linear(num_units, 1, bias=False)

    def _score(self, query, keys):
        # Put the query and the keys into Dense layer
        processed_query = self.query_layer(query)
        values = self.memory_layer(keys)

        # since the sizes of processed_query i [B x embedding],
        # we can't directly add it with the keys. therefore, we need
        # to add extra dimension, so the dimension of the query
        # now become [B x 1 x alignment unit size]
        extended_query = processed_query.unsqueeze(1)

        # The original formula is v * tanh(extended_query + values).
        # We can just use Dense layer and feed tanh as the input
        alignment = self.alignment_layer(F.tanh(extended_query + values))

        # Now the alignment size is [B x S x 1], We need to squeeze it
        # so that we can use Softmax later on. Converting to [B x S]
        return alignment.squeeze(2)

    def forward(self, query, keys):
        # Calculate the alignment score
        alignment_score = self._score(query, keys)

        # Put it into softmax to get the weight of every steps
        weight = F.softmax(alignment_score, dim=-1)

        # To get the context, this is the original formula
        # context = sum(weight * keys)
        # In order to multiply those two, we need to reshape the weight
        # from [B x S] into [B x S x 1] for broacasting.
        # The multiplication will result in [B x S x embedding]. Remember,
        # we want the score as the sum over all the steps. Therefore, we will
        # sum it over the 1st index
        context = weight.unsqueeze(2) * keys
        total_context = context.sum(1)

        return total_context, alignment_score


class LuongAttention(nn.Module):
    _SCORE_FN = {
        "dot": "_dot_score",
        "general": "_general_score",
        "concat": "_concat_score"
    }

    def __init__(self,
                 attention_window_size,
                 num_units,
                 query_size,
                 memory_size,
                 alignment="local",
                 score_fn="dot"):
        super(LuongAttention, self).__init__()

        if score_fn not in self._SCORE_FN.keys():
            raise ValueError()

        self._attention_window_size = attention_window_size
        self._softmax = nn.Softmax()
        self._score_fn = score_fn
        self._alignment = alignment

        self.query_layer = nn.Linear(query_size, num_units, bias=False)
        self.predictive_alignment_layer = nn.Linear(num_units, 1, bias=False)
        self.alignment_layer = nn.Linear(num_units, 1, bias=False)

        if score_fn == "general":
            self.general_memory_layer = nn.Linear(
                memory_size, query_size, bias=False)
        elif score_fn == "concat":
            self.concat_memory_layer1 = nn.Linear(
                2 * memory_size, num_units, bias=False)
            self.concat_memory_layer2 = nn.Linear(num_units, 1, bias=False)

    def _dot_score(self, query, keys):
        depth = query.size(-1)
        key_units = keys.size(-1)
        if depth != key_units:
            raise ValueError(
                "Incompatible inner dimensions between query and keys. "
                "Query has units: %d. Keys have units: %d. "
                "Dot score requires you to have same size between num_units in "
                "query and keys" % (depth, key_units))

        # Expand query to [B x 1 x embedding dim] for broadcasting
        extended_query = query.unsqueeze(1)

        # Transpose the keys so that we can multiply it
        tkeys = keys.transpose(1, 2)

        alignment = torch.matmul(extended_query, tkeys)

        # Result of the multiplication will be in size [B x 1 x embedding dim]
        # we can safely squeeze the dimension
        return alignment.squeeze(1)

    def _general_score(self, query, keys):
        weighted_keys = self.general_memory_layer(keys)
        extended_query = query.unsqueeze(1)
        weighted_keys = weighted_keys.transpose(1, 2)

        alignment = torch.matmul(extended_query, weighted_keys)
        return alignment.squeeze(1)

    def _concat_score(self, query, keys):
        expanded_query = query.unsqueeze(1).expand(*keys.size())
        concatenated_hidden = torch.cat([expanded_query, keys], dim=2)
        weighted_concatenated_hidden = self.concat_memory_layer1(
            concatenated_hidden)
        temp_score = F.tanh(weighted_concatenated_hidden)
        alignment = self.concat_memory_layer2(temp_score)

        return alignment.squeeze(2)

    def forward(self, query, keys, key_lengths):
        score_fn = getattr(self, self._SCORE_FN[self._score_fn])
        alignment_score = score_fn(query, keys)

        weight = F.softmax(alignment_score, dim=-1)

        if self._alignment == "local":
            extended_key_lengths = key_lengths.unsqueeze(1)
            preprocessed_query = self.query_layer(query)

            activated_query = F.tanh(preprocessed_query)
            sigmoid_query = F.sigmoid(
                self.predictive_alignment_layer(activated_query))
            predictive_alignment = extended_key_lengths * sigmoid_query

            ai_start = predictive_alignment - self._attention_window_size
            ai_end = predictive_alignment + self._attention_window_size

            std = torch.FloatTensor([self._attention_window_size / 2.]).pow(2)
            alignment_point_dist = (
                    extended_key_lengths - predictive_alignment).pow(2)

            alignment_point_dist = (
                -(alignment_point_dist / (2 * std[0]))).exp()
            weight = weight * alignment_point_dist

            contexts = []
            for i in range(weight.size(0)):
                start = ai_start[i].int().data.numpy()[0]
                end = ai_end[i].int().data.numpy()[0]

                aligned_weight = weight[i, start:end]
                aligned_keys = keys[i, start:end]

                aligned_context = aligned_weight.unsqueeze(1) * aligned_keys
                contexts.append(aligned_context.sum(0))

            total_context = torch.stack(contexts, 0)
        elif self._alignment == "global":
            context = weight.unsqueeze(2) * keys
            total_context = context.sum(1)

        return total_context, alignment_score

    @property
    def attention_window_size(self):
        return self._attention_window_size


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 d_model,
                 n_head=8,
                 dropout_p=0.5):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_head == 0, 'd_model must be dividable by n_head'

        self.d_model = d_model 
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.scale_factor = 1 / math.sqrt(d_model)
        # self._key_dim = torch.tensor(key_dim, requires_grad=False).float()

        self.query_layer = nn.Linear(d_model, d_model, bias=False)
        self.key_layer = nn.Linear(d_model, d_model, bias=False)
        self.value_layer = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout_p)

    def forward(self, query, keys, mask=None):
        # inputs: v_tilde_l (last layer's vertex representation)
        # query: (n_node, 1, d_hidden); v_i
        # keys: (n_node, n_node, d_hidden); v_j
        # mask: (n_node, 1, n_node); 1 for unconnected nodes
        Q = self.query_layer(query).cuda() # TODO: query, key를 미리 cuda로 보내놓기?
        K = self.key_layer(keys).cuda()
        V = self.value_layer(keys).cuda()

        # split each Q, K and V into h different values from dim 2
        # and then merge them back together in dim 0
        # K: (n_node, n_node, d_hidden) => (n_node*n_head, n_node, d_hidden/n_head)
        # e.g. K: (27, 27, 500) => (108, 27, 125) with 4 heads
        Q = torch.cat(Q.split(split_size=self.d_head, dim=-1), dim=0)
        K = torch.cat(K.split(split_size=self.d_head, dim=-1), dim=0)
        V = torch.cat(V.split(split_size=self.d_head, dim=-1), dim=0)

        attn = torch.matmul(Q, K.transpose(1, 2)) # (n_node*n_head, 1, n_node) e.g. (108, 1, 27)
        attn = attn * self.scale_factor

        if mask is not None:
            # for graph attention: mask out nodes that are unconnected with the query node 
            mask = mask.repeat(self.n_head, 1, 1)  # e.g. mask: (27, 1, 27) => (108, 1, 27)
            attn.masked_fill_(mask, -float('inf'))

        ## compute alpha (eq.2)
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        ## compute v_caret (eq.1)
        attn = torch.matmul(attn, V) # e.g. (108, 1, 125)
        # convert attention back to its input original size
        n_node = int(attn.size(0) / self.n_head) 
        attn = torch.cat(attn.split(split_size=n_node, dim=0), dim=-1)
        # residual connection
        attn += query # e.g. (27, 1, 500)

        # apply batch normalization
        # attn = self.bn(attn.transpose(1, 2)).transpose(1, 2)
        # apply layer normalization
        # attn = self.ln(attn)

        return attn
