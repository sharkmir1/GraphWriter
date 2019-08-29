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

    def forward(self, batch):
        if self.args.title:
            title, _ = self.title_encoder(batch.src) # (bsz, max_title_len, d_hidden)
            title_mask = self.mask_from_list(title.size(), batch.src[1]).unsqueeze(1) # (bsz, 1, max_title_len)
        
        # abstract with all tag indices removed, goes into decoder
        outp, _ = batch.out # (bsz, max_tgt_len)

        # refer to batchify in dataset.py
        ents = batch.ent  # tuple of (ent, phlens, elens) 
        ent_lens = ents[2]
        # encoded hidden state of entities in b
        ents = self.ent_encoder(ents) # (bsz, max_n_entity, d_hidden) 

        if self.is_graph:
            # batch.rel[0]: list (len: bsz) of adjacency matrices, each (N, N); N = n_entity + 1(for global node) + 2*n_rel
            # batch.rel[1]: list (len: bsz) of relations, each (1 + 2*n_rel)
            glob, nodes, node_mask = self.graph_encoder(batch.rel[0], batch.rel[1], (ents, ent_lens))

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
            plan_logits = self.splan(hx, nodes, node_mask.clone(), ent_lens, batch.sordertgt)
            schange = (outp == self.args.dottok).t()
            node_mask.fill_(0)
            planplace = torch.zeros(hx.size(0)).long()
            for i, m in enumerate(batch.sorder):
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
                        if planplace[j] < len(batch.sorder[j]):
                            node_mask[j] = 0
                            m = batch.sorder[j][planplace[j]]
                            node_mask[j][0][batch.sorder[j][planplace[j]]] = 1

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
        
        # copy distribution; attention of each abstract word on each entity
        _, a_copy = self.mattn(outputs, (ents, ent_lens)) # (bsz, max_abstract_len, max_n_entity)
        dist_copy = p_copy * a_copy
        
        # generation distribution
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
        mask = outp >= self.args.ntoks  # 1 if words are indexed tags (e.g. <method_0>)
        if mask.sum() > 0:
            idxs = (outp - self.args.ntoks)
            idxs = idxs[mask]
            verts = vertex.index_select(1, idxs)
            outp.masked_scatter_(mask, verts)

        # if words all in output vocab, return same outp
        return outp

    def beam_generate(self, batch, width, k, max_len=200):
        # k: top-k

        if self.args.title:
            title, _ = self.title_encoder(batch.src) # (1, title_len, d_hidden)
            title_mask = self.mask_from_list(title.size(), batch.src[1]).unsqueeze(1)  # (1, 1, title_len)

        ents = batch.ent # tuple of (ent, phlens, elens)
        ent_lens = ents[2]
        ents = self.ent_encoder(ents) # (1, entity num, d_hidden)

        if self.is_graph:
            glob, nodes, node_mask = self.graph_encoder(batch.rel[0], batch.rel[1], (ents, ent_lens))
            hx = glob
            node_mask = (node_mask == 0).unsqueeze(1)
        else:
            node_mask = self.mask_from_list(ents.size(), ent_lens).unsqueeze(1)
            hx = ents.mean(dim=1)[0]
            nodes = ents

        if self.args.plan:
            plan_logits = self.splan.plan_decode(hx, nodes, node_mask.clone(), ent_lens)
            print(plan_logits.size())
            sorder = ' '.join([str(x) for x in plan_logits.max(1)[1][0].tolist()])
            print(sorder)
            sorder = [x.strip() for x in sorder.split("-1")]
            sorder = [[int(y) for y in x.strip().split(" ")] for x in sorder]
            node_mask.fill_(0)
            planplace = torch.zeros(hx.size(0)).long()
            for i, m in enumerate(sorder):
                node_mask[i][0][m[0]] = 1
        else:
            plan_logits = None

        cx = hx.clone() # (beam size, d_hidden)

        context = self.graph_attn(hx.unsqueeze(1), nodes, mask=node_mask).squeeze(1) # (1, d_hidden)

        if self.args.title:
            context_title = self.title_attn(hx.unsqueeze(1), title, mask=title_mask).squeeze(1) # (1, d_hidden)
            context = torch.cat((context, context_title), 1) # (beam size, d_hidden*2)
        
        outp = torch.LongTensor(ents.size(0), 1).fill_(self.starttok).cuda() # initially, (1, 1) / start token
        beam = None

        # TODO: self.maxlen, self.starttok, self.endtok, self.eostok 등을 generator.py에서 만드는데 그렇게 하지 말고 인자로 넣어줄 것.
        for i in range(max_len):
            op = self.emb_w_vertex(outp.clone(), batch.nerd)
            # tag로부터 index를 없애는 작업. train할 때도 not indexed => indexed 로 train 되었기 때문.
            
            if self.args.plan:
                schange = op == self.args.dottok
                if schange.nonzero().size(0) > 0:
                    print(schange, planplace, sorder)
                    planplace[schange.nonzero().squeeze()] += 1
                    for j in schange.nonzero().squeeze(1):
                        if planplace[j] < len(sorder[j]):
                            node_mask[j] = 0
                            m = sorder[j][planplace[j]]
                            node_mask[j][0][sorder[j][planplace[j]]] = 1
            
            op = self.emb(op).squeeze(1)  # (beam size, 500)
            prev = torch.cat((a, op), 1)  # (beam size, 1000)
            hx, cx = self.lstm(prev, (hx, cx))

            context = self.graph_attn(hx.unsqueeze(1), nodes, mask=node_mask).squeeze(1)

            if self.args.title:
                context_title = self.title_attn(hx.unsqueeze(1), title, mask=title_mask).squeeze(1)
                context = torch.cat((context, context_title), 1) # (beam size, d_hidden*2)

            outputs = torch.cat((hx, context), 1).unsqueeze(1) # (beam size, 1, d_hidden*3)
            p_copy = torch.sigmoid(self.switch(outputs)) # (beam size, 1, 1)

            a_gen = self.out(outputs) # (beam size, 1, n_tgt_vocab)
            a_gen = torch.softmax(a_gen, -1)
            dist_gen = (1 - p_copy) * a_gen

            _, a_copy = self.mattn(outputs, (ents, ent_lens))
            dist_copy = p_copy * a_copy

            pred = torch.cat([dist_copy, dist_gen], -1) # (beam size, 1, target vocab size + entity num)
            # remove probability for special tokens <unk>, <init>
            pred[:, :, 0].fill_(0) 
            pred[:, :, 1].fill_(0)

            # TODO: 여기부터 다시 고칠것
            pred = pred + (1e-6 * torch.ones_like(pred))
            decoded = pred.log()
            scores, words = decoded.topk(dim=2, k=k)  # (beam size, 1, k), (beam size, 1, k)
            if not beam:
                beam = Beam(words.squeeze(), scores.squeeze(), [hx for i in range(width)],
                            [cx for i in range(width)], [a for i in range(width)], width, k, self.args.ntoks)
                beam.endtok = self.endtok
                beam.eostok = self.eostok
                nodes = nodes.repeat(len(beam.beam), 1, 1)  # (beam size, adjacency matrix len, 500)
                mask = mask.repeat(len(beam.beam), 1, 1)  # (beam size, 1, adjacency matrix len) => all 0?
                
                if self.args.title:
                    title = title.repeat(len(beam.beam), 1, 1) # (beam size, title_len, d_hidden)
                    title_mask = title_mask.repeat(len(beam.beam), 1, 1) # (1, 1, title_len)
                
                if self.args.plan:
                    planplace = planplace.unsqueeze(0).repeat(len(beam.beam), 1)
                    sorder = sorder * len(beam.beam)

                ents = ents.repeat(len(beam.beam), 1, 1) # (beam size, entity num, d_hidden)
                ent_lens = ent_lens.repeat(len(beam.beam)) # (beam_size)
            else:
                if not beam.update(scores, words, hx, cx, a):
                    break
                # if beam size changes (i.e. any of beam ends), change size of weight matrices accordingly
                nodes = nodes[:len(beam.beam)]
                mask = mask[:len(beam.beam)]
                if self.args.title:
                    title = title[:len(beam.beam)]
                    title_mask = title_mask[:len(beam.beam)]
                if self.args.plan:
                    planplace = planplace[:len(beam.beam)]
                    sorder = sorder[0] * len(beam.beam)
                ents = ents[:len(beam.beam)]
                ent_lens = ent_lens[:len(beam.beam)]
            outp = beam.getwords()  # (beam size,) / next word for each beam
            hx = beam.geth()  # (beam size, 500)
            cx = beam.getc()  # (beam size, 500)
            a = beam.getlast()  # (beam size, 1000)

        return beam
