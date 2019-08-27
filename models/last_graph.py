import torch
import math
from torch import nn
from torch.nn import functional as F
from models.graphAttn import GAT
from models.attention import MultiHeadAttention

import ipdb


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Block(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attn = MultiHeadAttention(args.hsz, args.hsz, args.hsz, h=args.heads, dropout_p=args.blockdrop)
        self.l1 = nn.Linear(args.hsz, args.hsz * args.heads)
        self.l2 = nn.Linear(args.hsz * args.heads, args.hsz)
        self.ln_1 = nn.LayerNorm(args.hsz)
        self.ln_2 = nn.LayerNorm(args.hsz)
        self.drop = nn.Dropout(args.drop)
        # self.act = gelu
        self.act = nn.PReLU(args.hsz * args.heads)
        self.gat_act = nn.PReLU(args.hsz)

    def forward(self, q, k, m):
        out = self.attn(q, k, mask=m).squeeze(1)  # (adj_len(= # of entities + # of relations), d_hidden)
        t = self.ln_1(out)
        # FFN
        out = self.drop(self.l2(self.act(self.l1(t))))
        out = self.ln_2(out + t)
        return out  # V_tilde in paper / (adj_len ( = # of entities + # of relations), d_hidden)


class graph_encode(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.renc = nn.Embedding(args.rtoks, args.hsz)
        nn.init.xavier_normal_(self.renc.weight)

        if args.model == "gat":
            self.gat = nn.ModuleList(
                [MultiHeadAttention(args.hsz, args.hsz, args.hsz, h=4, dropout_p=args.blockdrop) for _ in range(args.prop)])
        elif args.model == "graph":
            self.gat = nn.ModuleList([Block(args) for _ in range(args.prop)])
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
                 glob: global node embedding (batch_size, d_hidden)
                 gents: (batch_size, max n_node, d_hidden)
                 emask: (batch_size, max n_node) / 각 adj_matrix의 size보다 작은 부분만 1
        """
        ents, ent_lens = ents
        if self.args.entdetach:
            ents = torch.tensor(ents, requires_grad=False)
        rels = [self.renc(x) for x in rels]  # rels: list of (n_relation, d_embed) per row in batch
        glob = []
        graphs = []

        for i, adj in enumerate(adjs):
            # Concat entity embeddings and relation embeddings
            q_graph = torch.cat((ents[i][:ent_lens[i]], rels[i]), 0) # (n_entity + n_relation, d_hidden)
            # number of nodes = n_entity + n_relation
            N = q_graph.size(0)

            if self.sparse:
                lens = [len(x) for x in adj]
                m = max(lens)
                mask = torch.arange(0, m).unsqueeze(0).repeat(len(lens), 1).long()
                mask = (mask <= torch.LongTensor(lens).unsqueeze(1)).cuda()
                mask = (mask == 0).unsqueeze(1)
            else:
                mask = (adj == 0).unsqueeze(1) # (N, 1, N)

            for j in range(self.prop):
                if self.sparse:
                    kv_graph = [q_graph[k] for k in adj]
                    kv_graph = [self.pad(x, m) for x in kv_graph]
                    kv_graph = torch.stack(kv_graph, 0)
                    # print(kv_graph.size(),q_graph.size(),mask.size())
                    q_graph = self.gat[j](q_graph.unsqueeze(1), kv_graph, mask)
                else:
                    # Repeat q_graph N times
                    kv_graph = torch.tensor(q_graph.repeat(N, 1).view(N, N, -1), requires_grad=False) # (N, N, d_hidden)

                    # TF block
                    q_graph = self.gat[j](q_graph.unsqueeze(1), kv_graph, mask)  # (N, d_hidden)
                    if self.args.model == 'gat':
                        q_graph = q_graph.squeeze(1)
                        q_graph = self.gat_act(q_graph)

            graphs.append(q_graph)
            # global context node @ diag of adjacency matrix
            glob.append(q_graph[ent_lens[i]]) # append each global node's embedding

        # graphs: list (len: bsz), each (n_node, d_hidden)
        node_lens = [x.size(0) for x in graphs]
        nodes = [self.pad(x, max(node_lens)) for x in graphs]
        nodes = torch.stack(nodes, 0)   # (batch_size, max n_node, d_hidden)
        node_lens = torch.LongTensor(node_lens)
        node_mask = torch.arange(0, nodes.size(1)).unsqueeze(0).repeat(nodes.size(0), 1).long()
        node_mask = (node_mask <= node_lens.unsqueeze(1)).cuda()  # (batch_size, max adj_matrix size) / 각 adj_matrix의 size보다 작은 부분만 1
        glob = torch.stack(glob, 0)  # (batch_size, d_hidden)
        return glob, nodes, node_mask
