import torch
from torch import nn
from models.attention import MultiHeadAttention, MatrixAttn
from models.list_encoder import list_encode, lseq_encode
from models.last_graph import graph_encode
from models.beam import Beam
from models.splan import splanner
import ipdb


class model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        cattimes = 3 if args.title else 2
        self.emb = nn.Embedding(args.ntoks, args.hsz)
        self.lstm = nn.LSTMCell(args.hsz * cattimes, args.hsz)
        self.out = nn.Linear(args.hsz * cattimes, args.tgttoks)
        self.ent_encoder = list_encode(args)
        self.ent_out = nn.Linear(args.hsz, 1)
        self.switch = nn.Linear(args.hsz * cattimes, 1)
        self.attn_graph = MultiHeadAttention(args.hsz, args.hsz, args.hsz, h=args.heads, dropout_p=args.drop)
        self.mattn = MatrixAttn(args.hsz * cattimes, args.hsz)
        self.is_graph = (args.model in ['graph', 'gat', 'gtrans'])
        print(args.model)
        if self.is_graph:
            self.graph_encoder = graph_encode(args) # graph transformer
        if args.plan:
            self.splan = splanner(args)
        if args.title:
            self.title_encoder = lseq_encode(args, toks=args.ninput)  # title encoder
            self.attn_title = MultiHeadAttention(args.hsz, args.hsz, args.hsz, h=args.heads, dropout_p=args.drop)
            self.mix = nn.Linear(args.hsz, 1)

    def forward(self, b):
        if self.args.title:
            title_encoded, _ = self.title_encoder(b.src)
            title_mask = self.mask_from_list(title_encoded.size(), b.src[1]).unsqueeze(1) # (bsz, 1, max_title_len)
        outp, _ = b.out # target abstract
        ents = b.ent  # (ent, phlens, elens) => refer to fixBatch in lastDataset.py
        entlens = ents[2]
        ents = self.ent_encoder(ents)
        # ents: (batch_size (num of rows), max entity num, 500) / encoded hidden state of entities in batch

        if self.is_graph:
            # b.rel[0]: list (len: bsz) of adjacency matrices, each (N, N); N = n_entity + 1(global) + 2*n_rel
            # b.rel[1]: list (len: bsz) of relations, each (1 + 2*n_rel)
            glob, keys, mask = self.graph_encoder(b.rel[0], b.rel[1], (ents, entlens)) # glob, nodes, node_mask
            hx = glob
            mask = mask == 0
        else:
            mask = self.mask_from_list(ents.size(), entlens)
            hx = ents.mean(dim=1)
            keys = ents
        mask = mask.unsqueeze(1)
        if self.args.plan:
            planlogits = self.splan(hx, keys, mask.clone(), entlens, b.sordertgt)
            schange = (outp == self.args.dottok).t()
            mask.fill_(0)
            planplace = torch.zeros(hx.size(0)).long()
            for i, m in enumerate(b.sorder):
                mask[i][0][m[0]] = 1
        else:
            planlogits = None

        cx = torch.tensor(hx) # (bsz, d_hidden)
        a = torch.zeros_like(hx)

        if self.args.title:
            a_title = self.attn_title(hx.unsqueeze(1), title_encoded, mask=title_mask).squeeze(1)
            a = torch.cat((a, a_title), 1)

        ### Decoding
        e = self.emb(outp).transpose(0, 1) # (max_abstract_len, bsz, d_hidden)
        outputs = []
        for i, k in enumerate(e):
            # k = self.emb(k)
            if self.args.plan:
                if schange[i].nonzero().size(0) > 0:
                    planplace[schange[i].nonzero().squeeze()] += 1
                    for j in schange[i].nonzero().squeeze(1):
                        if planplace[j] < len(b.sorder[j]):
                            mask[j] = 0
                            m = b.sorder[j][planplace[j]]
                            mask[j][0][b.sorder[j][planplace[j]]] = 1
            prev = torch.cat((a, k), 1)
            hx, cx = self.lstm(prev, (hx, cx))

            # Attend to graph
            a = self.attn_graph(hx.unsqueeze(1), keys, mask=mask).squeeze(1)

            # Attend to title (optional)
            if self.args.title:
                a_title = self.attn_title(hx.unsqueeze(1), title_encoded, mask=title_mask).squeeze(1)
                # a =  a + (self.mix(hx)*a_title)
                a = torch.cat((a, a_title), 1)

            out = torch.cat((hx, a), 1)
            outputs.append(out)

        l = torch.stack(outputs, 1)
        s = torch.sigmoid(self.switch(l))
        o = self.out(l)
        o = torch.softmax(o, 2)
        o = s * o
        # compute copy attn
        _, z = self.mattn(l, (ents, entlens))
        # z = torch.softmax(z,2)
        z = (1 - s) * z
        o = torch.cat((o, z), 2)
        o = o + (1e-6 * torch.ones_like(o))
        return o.log(), z, planlogits

    def mask_from_list(self, size, l):
        mask = torch.arange(0, size[1]).unsqueeze(0).repeat(size[0], 1).long().cuda()
        mask = (mask <= l.unsqueeze(1))
        mask = mask == 0
        return mask

    def emb_w_vertex(self, outp, vertex):
        mask = outp >= self.args.ntoks
        if mask.sum() > 0:
            idxs = (outp - self.args.ntoks)
            idxs = idxs[mask]
            verts = vertex.index_select(1, idxs)
            outp.masked_scatter_(mask, verts)

        return outp

    def beam_generate(self, b, beamsz, k):
        if self.args.title:
            title_encoded, _ = self.title_encoder(b.src)
            title_mask = self.mask_from_list(title_encoded.size(), b.src[1]).unsqueeze(1)
        ents = b.ent
        entlens = ents[2]
        ents = self.ent_encoder(ents)
        if self.is_graph:
            graph_ents, glob, graph_rels = self.graph_encoder(b.rel[0], b.rel[1], (ents, entlens))
            hx = glob
            # hx = ents.max(dim=1)[0]
            keys, mask = graph_rels
            mask = mask == 0
        else:
            mask = self.mask_from_list(ents.size(), entlens)
            hx = ents.max(dim=1)[0]
            keys = ents
        mask = mask.unsqueeze(1)
        if self.args.plan:
            planlogits = self.splan.plan_decode(hx, keys, mask.clone(), entlens)
            print(planlogits.size())
            sorder = ' '.join([str(x) for x in planlogits.max(1)[1][0].tolist()])
            print(sorder)
            sorder = [x.strip() for x in sorder.split("-1")]
            sorder = [[int(y) for y in x.strip().split(" ")] for x in sorder]
            mask.fill_(0)
            planplace = torch.zeros(hx.size(0)).long()
            for i, m in enumerate(sorder):
                mask[i][0][m[0]] = 1
        else:
            planlogits = None

        cx = torch.tensor(hx)
        a = self.attn_graph(hx.unsqueeze(1), keys, mask=mask).squeeze(1)
        if self.args.title:
            a_title = self.attn_title(hx.unsqueeze(1), title_encoded, mask=title_mask).squeeze(1)
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
            a = self.attn_graph(hx.unsqueeze(1), keys, mask=mask).squeeze(1)
            if self.args.title:
                a_title = self.attn_title(hx.unsqueeze(1), title_encoded, mask=title_mask).squeeze(1)
                # a =  a + (self.mix(hx)*a_title)
                a = torch.cat((a, a_title), 1)
            l = torch.cat((hx, a), 1).unsqueeze(1)
            s = torch.sigmoid(self.switch(l))
            o = self.out(l)
            o = torch.softmax(o, 2)
            o = s * o
            # compute copy attn
            _, z = self.mattn(l, (ents, entlens))
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
                    title_encoded = title_encoded.repeat(len(beam.beam), 1, 1)
                    title_mask = title_mask.repeat(len(beam.beam), 1, 1)
                if self.args.plan:
                    planplace = planplace.unsqueeze(0).repeat(len(beam.beam), 1)
                    sorder = sorder * len(beam.beam)

                ents = ents.repeat(len(beam.beam), 1, 1)
                entlens = entlens.repeat(len(beam.beam))
            else:
                if not beam.update(scores, words, hx, cx, a):
                    break
                keys = keys[:len(beam.beam)]
                mask = mask[:len(beam.beam)]
                if self.args.title:
                    title_encoded = title_encoded[:len(beam.beam)]
                    title_mask = title_mask[:len(beam.beam)]
                if self.args.plan:
                    planplace = planplace[:len(beam.beam)]
                    sorder = sorder[0] * len(beam.beam)
                ents = ents[:len(beam.beam)]
                entlens = entlens[:len(beam.beam)]
            outp = beam.getwords()
            hx = beam.geth()
            cx = beam.getc()
            a = beam.getlast()

        return beam
