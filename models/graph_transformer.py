import torch
import math
from torch import nn
from torch.nn import functional as F
from models.graphAttn import GAT
from models.attention import MultiHeadAttention

import ipdb


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attn = MultiHeadAttention(args.hsz, n_head=args.heads, dropout_p=args.blockdrop)
        self.ffn = nn.Sequential(
            nn.Linear(args.hsz, args.hsz * args.heads),
            nn.PReLU(args.hsz * args.heads),
            nn.Linear(args.hsz * args.heads, args.hsz)
        )
        self.ln_1 = nn.LayerNorm(args.hsz)
        self.ln_2 = nn.LayerNorm(args.hsz)
        self.drop = nn.Dropout(args.drop)
        self.gat_act = nn.PReLU(args.hsz)

    def forward(self, q, k, m):
        # v_caret
        out = self.attn(q, k, mask=m).squeeze(1) # (adj_len(= # of entities + # of relations), d_hidden)
        out = self.ln_1(out)
        residual = out.clone()
        out = self.ffn(out)
        out = self.drop(out) 
        out = self.ln_2(out + residual)
        
        # v_tilde
        return out # (adj_len ( = # of entities + # of relations), d_hidden)


class graph_encode(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.renc = nn.Embedding(args.rtoks, args.hsz)
        nn.init.xavier_normal_(self.renc.weight)

        if args.model == "gat":
            self.gat = nn.ModuleList([MultiHeadAttention(args.hsz, n_head=args.heads, dropout_p=args.blockdrop) for _ in range(args.prop)])
        elif args.model == "graph":
            self.gat = nn.ModuleList([TransformerBlock(args) for _ in range(args.prop)])
        else:
            raise NotImplementedError
        self.prop = args.prop # number of transformer blocks (layers)
        self.sparse = args.sparse

    def pad(self, tensor, length):
        return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).fill_(0)])

    def forward(self, adjs, rels, ents):
        """
        :param adjs: list of adjacency matrices in batch
        :param rels: list of tensors of relations per row in batch
        :param ents: tuple (tensor(batch_size, max entity num, d_hidden): encoded entities per row,
                            tensor(batch_size): # of entities in rows)
        :return: None, glob, (gents, emask)
                 glob: (batch_size, 500) / global node embedding
                 gents: (batch_size, max adj_matrix size, 500) / graph contextualized vertex encodings
                 emask: (batch_size, max adj_matrix size) / 각 adj_matrix의 size + 1보다 작은 부분만 1
        """
        ents, ent_lens = ents
        if self.args.entdetach:
            ents = ents.detach()
            # ents = torch.tensor(ents, requires_grad=False)

        rels = [self.renc(x) for x in rels]  # rels: list of (n_relation, d_embed) per row in batch
        globs = []
        graphs = []

        for i, adj in enumerate(adjs):
            # Concat entity embeddings and relation embeddings
            graph_q = torch.cat((ents[i][:ent_lens[i]], rels[i]), 0) # (n_node, d_hidden)
            
            # adj_matrix_size = n_node = n_entity + n_relation
            n_node = graph_q.size(0)

            ## Compute mask=1 for unconnected nodes
            if self.sparse:
                lens = [len(x) for x in adj]
                m = max(lens)
                mask = torch.arange(0, m).unsqueeze(0).repeat(len(lens), 1).long()
                mask = (mask <= torch.LongTensor(lens).unsqueeze(1)).cuda()
                mask = (mask == 0).unsqueeze(1)
            else:
                mask = (adj == 0).unsqueeze(1) # (n_node, 1, n_node)

            ## Pass through TF blocks
            for l in range(self.prop):
                if self.sparse:
                    graph_kv = [graph_q[k] for k in adj]
                    graph_kv = [self.pad(x, m) for x in graph_kv]
                    graph_kv = torch.stack(graph_kv, 0)
                    graph_q = self.gat[l](graph_q.unsqueeze(1), graph_kv, mask)
                else:
                    graph_kv = torch.tensor(graph_q.repeat(n_node, 1).view(n_node, n_node, -1), requires_grad=False) # (n_node, n_node, d_hidden)
                    graph_q = self.gat[l](graph_q.unsqueeze(1), graph_kv, mask) # (n_node, d_hidden)
                    if self.args.model == 'gat':
                        graph_q = graph_q.squeeze(1)
                        graph_q = self.gat_act(graph_q)

            graphs.append(graph_q)

            # global context node @ diag of adjacency matrix
            globs.append(graph_q[ent_lens[i]]) # append each global node's embedding

        # graphs: list (len: bsz), each (n_node, d_hidden)
        node_lens = [x.size(0) for x in graphs]
        nodes = [self.pad(x, max(node_lens)) for x in graphs]
        nodes = torch.stack(nodes, 0)   # (bsz, max n_node, d_hidden)
        node_lens = torch.LongTensor(node_lens)
        node_mask = torch.arange(0, nodes.size(1)).unsqueeze(0).repeat(nodes.size(0), 1).long()
        node_mask = (node_mask <= node_lens.unsqueeze(1)).cuda() # (bsz, max adj_matrix size) / 각 adj_matrix의 size보다 작은 부분만 1
        glob = torch.stack(globs, 0) # (bsz, d_hidden)
        return glob, nodes, node_mask
