import torch
from torch import nn
from models.attention import MultiHeadAttention, MatrixAttention
from models.list_encoder import list_encode, lseq_encode
from models.graph_transformer import graph_encode
from models.beam import Beam
from models.splan import splanner
import ipdb


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        n_attn_sources = 3 if args.title else 2
        self.emb = nn.Embedding(args.ntoks, args.hsz)
        self.ent_encoder = list_encode(args) # entity encoder
        self.is_graph = (args.model in ['graph', 'gat', 'gtrans']) # TODO; 필요하긴 한가
        if self.is_graph:
            self.graph_encoder = graph_encode(args) # graph transformer
            self.graph_attn = MultiHeadAttention(args.hsz, n_head=args.heads, dropout_p=args.drop) # graph transformer 내부에 있는 graph self-attention과는 별개.
        if args.title:
            self.title_encoder = lseq_encode(args, toks=args.ninput)
            self.title_attn = MultiHeadAttention(args.hsz, n_head=args.heads, dropout_p=args.drop)
            # self.mix = nn.Linear(args.hsz, 1)
        self.decoder = nn.LSTMCell(args.hsz * n_attn_sources, args.hsz)
        self.mattn = MatrixAttention(args.hsz * n_attn_sources, args.hsz) # makes fixed size context vector for all sources (title, graph, lstm hidden)
        self.switch = nn.Linear(args.hsz * n_attn_sources, 1)
        self.out = nn.Linear(args.hsz * n_attn_sources, args.n_tgt_vocab)
        if args.plan:
            self.splan = splanner(args)
        # print(args.model)

    def forward(self, b):
        if self.args.title:
            title, _ = self.title_encoder(b.src) # (bsz, max_title_len, d_hidden)
            title_mask = self.mask_from_list(title.size(), b.src[1]).unsqueeze(1) # (bsz, 1, max_title_len)
        
        # abstract with all tag indices removed, goes into decoder
        outp, _ = b.out # (bsz, max_tgt_len)

        # refer to fixBatch in lastDataset.py
        ents = b.ent  # tuple of (ent, phlens, elens) 
        ent_lens = ents[2]
        ents = self.ent_encoder(ents) # (bsz, max_n_entity, d_hidden) 

        if self.is_graph:
            # b.rel[0]: list (len: bsz) of adjacency matrices, each (N, N); N = n_entity + 1(for global node) + 2*n_rel
            # b.rel[1]: list (len: bsz) of relations, each (1 + 2*n_rel)
            glob, nodes, node_mask = self.graph_encoder(b.rel[0], b.rel[1], (ents, ent_lens))

            # initial hidden state for the decoding lstm
            hx = glob
            # 1 on relation vertices, else 0
            node_mask = (node_mask == 0).unsqueeze(1) # (bsz, 1, max_adj_matrix_size)
        else:
            node_mask = self.mask_from_list(ents.size(), ent_lens).unsqueeze(1)
            hx = ents.mean(dim=1)
            nodes = ents
        
        # TODO: ??
        if self.args.plan:
            plan_logits = self.splan(hx, nodes, node_mask.clone(), ent_lens, b.sordertgt)
            schange = (outp == self.args.dottok).t()
            node_mask.fill_(0)
            planplace = torch.zeros(hx.size(0)).long()
            for i, m in enumerate(b.sorder):
                node_mask[i][0][m[0]] = 1
        else:
            plan_logits = None

        cx = hx.clone() # (bsz, d_hidden)
        context = torch.zeros_like(hx)

        # TODO: 왜 여기에 한 번 더 있지? 아래에도 나오는데
        if self.args.title:
            context_title = self.title_attn(hx.unsqueeze(1), title, mask=title_mask).squeeze(1) # (bsz, d_hidden)
            context = torch.cat((context, context_title), 1) # (bsz, d_hidden*2)

        ### Decoding
        outp_embedded = self.emb(outp).transpose(0, 1) # (max_abstract_len, bsz, d_hidden)
        outputs = []
        for i, k in enumerate(outp_embedded):
            # k: (bsz, d_embed) / i_th keyword's embedding in the target abstract
            if self.args.plan:
                if schange[i].nonzero().size(0) > 0:
                    planplace[schange[i].nonzero().squeeze()] += 1
                    for j in schange[i].nonzero().squeeze(1):
                        if planplace[j] < len(b.sorder[j]):
                            node_mask[j] = 0
                            m = b.sorder[j][planplace[j]]
                            node_mask[j][0][b.sorder[j][planplace[j]]] = 1

            # TODO: d_embed should match d_hidden; else, make projection
            prev = torch.cat((context, k), 1) # (bsz, d_hidden*3)
            hx, cx = self.decoder(prev, (hx, cx)) # compute new hx, cx with current (hx, cx, input abstract word)

            # Attend to graph
            # context vector for each graph (c_g; equation 6,7 in paper)
            context = self.graph_attn(hx.unsqueeze(1), nodes, mask=node_mask).squeeze(1) # (bsz, d_hidden)

            # Attend to title (optional)
            if self.args.title:
                # context vector for each title (c_s in the paper)
                context_title = self.title_attn(hx.unsqueeze(1), title, mask=title_mask).squeeze(1)
                # final context vector c_t; concat with graph context vector
                context = torch.cat((context, context_title), 1)

            # [ h_t || c_t ] in paper
            out = torch.cat((hx, context), 1) # (bsz, d_hidden*3)
            outputs.append(out)

        outputs = torch.stack(outputs, 1) # (bsz, max_abstract_len, d_hidden*3)


        ### Pointer-generator

        # switch probability 
        p_copy = torch.sigmoid(self.switch(outputs)) # (bsz, max_abstract_len, 1)
        
        # generation distribution; attention of each abstract word on each entity
        _, a_copy = self.mattn(outputs, (ents, ent_lens)) # (bsz, max_abstract_len, max_n_entity)
        dist_copy = p_copy * a_copy
        
        # copy distribution
        a_gen = self.out(outputs) # (bsz, max_abstract_len, n_vocab)
        a_gen = torch.softmax(a_gen, -1)
        dist_gen = (1 - p_copy) * a_gen

        # pred = p_copy * a_copy + (1 - p_copy) * a_gen
        pred = torch.cat([dist_copy, dist_gen], -1) # (bsz, max_abstract_len, max_n_entity + n_vocab)
        # TODO: mask?
        # TODO: OOV HANDLING; final_dist = dist_gen.scatter_add(2, oov, dist_copy)
        pred = pred + (torch.empty_like(pred).fill_(1e-6))
        return pred.log(), dist_copy, plan_logits

    def mask_from_list(self, size, l):
        """
        :param size: (batch_size, title_len, d_hidden)
        :param l: tensor of (batch_size,) / lengths of each example
        :return: (batch_size, 1, title_len) / 각 row의 길이 + 1보다 더 큰 부분만 1, else 0
        """
        mask = torch.arange(0, size[1]).unsqueeze(0).repeat(size[0], 1).long().cuda()
        mask = (mask <= l.unsqueeze(1))
        mask = mask == 0
        return mask

    def emb_w_vertex(self, outp, vertex):
        """
        e.g. assume outp: [[3, 100, 11744, 3]] / ntoks: 11738 / vertex: [~, ~, ~, ~, ~, ~, 11733, ~]
        11744 accords to word among [<method_6>, ... <task_6>] (i.e. it is 6th entity in this row),
        and in vertex, the 6th entity in this row accords to 11733. (which is <method>)

        then function returns [[3, 100, 11733, 3]] as output.

        :param outp: (beam_size,) / selected word for each beam (including indexed entities e.g. <method_0>)
        :param vertex: (1, num of entity in this row) /
                       indices of not-indexed entities (same as that in output vocab) + <eos>
        :return: output with all indexed tags changed to not-indexed tags (only types)
        """
        mask = outp >= self.args.ntoks
        if mask.sum() > 0:
            idxs = (outp - self.args.ntoks)
            idxs = idxs[mask]
            verts = vertex.index_select(1, idxs)
            outp.masked_scatter_(mask, verts)

        return outp

    def beam_generate(self, b, beamsz, k):
        if self.args.title:
            title, _ = self.title_encoder(b.src)
            title_mask = self.mask_from_list(title.size(), b.src[1]).unsqueeze(1)
        ents = b.ent
        ent_lens = ents[2]
        ents = self.ent_encoder(ents)
        if self.is_graph:
            graph_ents, glob, graph_rels = self.graph_encoder(b.rel[0], b.rel[1], (ents, ent_lens))
            hx = glob
            # hx = ents.max(dim=1)[0]
            keys, mask = graph_rels
            mask = mask == 0
        else:
            mask = self.mask_from_list(ents.size(), ent_lens)
            hx = ents.max(dim=1)[0]
            keys = ents
        mask = mask.unsqueeze(1)
        if self.args.plan:
            plan_logits = self.splan.plan_decode(hx, keys, mask.clone(), ent_lens)
            print(plan_logits.size())
            sorder = ' '.join([str(x) for x in plan_logits.max(1)[1][0].tolist()])
            print(sorder)
            sorder = [x.strip() for x in sorder.split("-1")]
            sorder = [[int(y) for y in x.strip().split(" ")] for x in sorder]
            mask.fill_(0)
            planplace = torch.zeros(hx.size(0)).long()
            for i, m in enumerate(sorder):
                mask[i][0][m[0]] = 1
        else:
            plan_logits = None

        cx = torch.tensor(hx)
        a = self.graph_attn(hx.unsqueeze(1), keys, mask=mask).squeeze(1)
        if self.args.title:
            a_title = self.title_attn(hx.unsqueeze(1), title, mask=title_mask).squeeze(1)
            a = torch.cat((a, a_title), 1)
        outputs = []
        outp = torch.LongTensor(ents.size(0), 1).fill_(self.starttok).cuda()
        beam = None
        for i in range(self.maxlen):
            op = self.emb_w_vertex(outp.clone(), b.nerd)
            if self.args.plan:
                schange = op == self.args.dottok
                if schange.nonzero().size(0) > 0:
                    print(schange, planplace, sorder)
                    planplace[schange.nonzero().squeeze()] += 1
                    for j in schange.nonzero().squeeze(1):
                        if planplace[j] < len(sorder[j]):
                            mask[j] = 0
                            m = sorder[j][planplace[j]]
                            mask[j][0][sorder[j][planplace[j]]] = 1
            op = self.emb(op).squeeze(1)
            prev = torch.cat((a, op), 1)
            hx, cx = self.lstm(prev, (hx, cx))
            a = self.graph_attn(hx.unsqueeze(1), keys, mask=mask).squeeze(1)
            if self.args.title:
                a_title = self.title_attn(hx.unsqueeze(1), title, mask=title_mask).squeeze(1)
                # a =  a + (self.mix(hx)*a_title)
                a = torch.cat((a, a_title), 1)
            l = torch.cat((hx, a), 1).unsqueeze(1)
            s = torch.sigmoid(self.switch(l))
            o = self.out(l)
            o = torch.softmax(o, 2)
            o = s * o
            # compute copy attn
            _, z = self.mattn(l, (ents, ent_lens))
            # z = torch.softmax(z,2)
            z = (1 - s) * z
            o = torch.cat((o, z), 2)
            o[:, :, 0].fill_(0)
            o[:, :, 1].fill_(0)
            '''
      if beam:
        for p,q in enumerate(beam.getPrevEnt()):
          o[p,:,q].fill_(0)
        for p,q in beam.getIsStart():
          for r in q:
            o[p,:,r].fill_(0)
      '''

            o = o + (1e-6 * torch.ones_like(o))
            decoded = o.log()
            scores, words = decoded.topk(dim=2, k=k)
            if not beam:
                beam = Beam(words.squeeze(), scores.squeeze(), [hx for i in range(beamsz)],
                            [cx for i in range(beamsz)], [a for i in range(beamsz)], beamsz, k, self.args.ntoks)
                beam.endtok = self.endtok
                beam.eostok = self.eostok
                keys = keys.repeat(len(beam.beam), 1, 1)
                mask = mask.repeat(len(beam.beam), 1, 1)
                if self.args.title:
                    title = title.repeat(len(beam.beam), 1, 1)
                    title_mask = title_mask.repeat(len(beam.beam), 1, 1)
                if self.args.plan:
                    planplace = planplace.unsqueeze(0).repeat(len(beam.beam), 1)
                    sorder = sorder * len(beam.beam)

                ents = ents.repeat(len(beam.beam), 1, 1)
                ent_lens = ent_lens.repeat(len(beam.beam))
            else:
                if not beam.update(scores, words, hx, cx, a):
                    break
                keys = keys[:len(beam.beam)]
                mask = mask[:len(beam.beam)]
                if self.args.title:
                    title = title[:len(beam.beam)]
                    title_mask = title_mask[:len(beam.beam)]
                if self.args.plan:
                    planplace = planplace[:len(beam.beam)]
                    sorder = sorder[0] * len(beam.beam)
                ents = ents[:len(beam.beam)]
                ent_lens = ent_lens[:len(beam.beam)]
            outp = beam.getwords()
            hx = beam.geth()
            cx = beam.getc()
            a = beam.getlast()

        return beam
