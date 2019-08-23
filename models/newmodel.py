import torch
from torch import nn
from models.attention import MultiHeadAttention, MatrixAttn
from models.list_encoder import list_encode, lseq_encode
from models.last_graph import graph_encode
from models.beam import Beam
from models.splan import splanner


class model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        cattimes = 3 if args.title else 2
        self.emb = nn.Embedding(args.ntoks, args.hsz)
        self.lstm = nn.LSTMCell(args.hsz * cattimes, args.hsz)
        self.out = nn.Linear(args.hsz * cattimes, args.tgttoks)
        self.le = list_encode(args)
        self.entout = nn.Linear(args.hsz, 1)
        self.switch = nn.Linear(args.hsz * cattimes, 1)
        self.attn = MultiHeadAttention(args.hsz, args.hsz, args.hsz, h=4, dropout_p=args.drop)
        self.mattn = MatrixAttn(args.hsz * cattimes, args.hsz)
        self.graph = (args.model in ['graph', 'gat', 'gtrans'])
        print(args.model)
        if self.graph:
            self.ge = graph_encode(args)
        if args.plan:
            self.splan = splanner(args)
        if args.title:
            self.tenc = lseq_encode(args, toks=args.ninput)  # title encoder
            self.attn2 = MultiHeadAttention(args.hsz, args.hsz, args.hsz, h=4, dropout_p=args.drop)
            # attn2: computes c_s (decoding-phase context vector for title)
            self.mix = nn.Linear(args.hsz, 1)

    def forward(self, b):
        if self.args.title:
            tencs, _ = self.tenc(b.src)  # (batch_size, title_len, 500)
            tmask = self.maskFromList(tencs.size(), b.src[1]).unsqueeze(1)  # (batch_size, 1, title_len)
        outp, _ = b.out  # (batch_size, max abstract len) / all tag indices removed
        ents = b.ent  # tuple of (ent, phlens, elens) / refer to fixBatch in lastDataset.py
        entlens = ents[2]
        ents = self.le(ents)
        # ents: (batch_size (num of rows), max entity num, 500) / encoded hidden state of entities in batch

        if self.graph:
            gents, glob, grels = self.ge(b.rel[0], b.rel[1], (ents, entlens))
            hx = glob  # this is the initial hidden state for decoding lstm
            keys, mask = grels
            mask = mask == 0  # (batch_size, max adj_matrix size) / 1 on relation vertices, else 0
        else:
            mask = self.maskFromList(ents.size(), entlens)
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

        cx = torch.tensor(hx)  # (batch_size, 500)
        a = torch.zeros_like(hx)
        if self.args.title:
            a2 = self.attn2(hx.unsqueeze(1), tencs, mask=tmask).squeeze(1)
            # a2: (batch_size, 500) / c_s, context vector for each title
            a = torch.cat((a, a2), 1)
            # a: (batch_size, 1000) / cat of c_g, c_s (c_g computed below)

        e = self.emb(outp).transpose(0, 1)  # (max abstract len, batch_size, 500)
        outputs = []
        for i, k in enumerate(e):
            # k: (batch_Size, 500) / i_th keyword's embedding in the target abstract
            if self.args.plan:
                if schange[i].nonzero().size(0) > 0:
                    planplace[schange[i].nonzero().squeeze()] += 1
                    for j in schange[i].nonzero().squeeze(1):
                        if planplace[j] < len(b.sorder[j]):
                            mask[j] = 0
                            m = b.sorder[j][planplace[j]]
                            mask[j][0][b.sorder[j][planplace[j]]] = 1
            prev = torch.cat((a, k), 1)  # (batch_size, 1500)
            hx, cx = self.lstm(prev, (hx, cx))  # compute new hx, cx with current (hx, cx, input abstract word)
            a = self.attn(hx.unsqueeze(1), keys, mask=mask).squeeze(1)
            # a: (batch_size, 500) / c_g, context vector for each graph (equation 6,7 in paper)
            if self.args.title:
                a2 = self.attn2(hx.unsqueeze(1), tencs, mask=tmask).squeeze(1)
                # a2: (batch_size, 500) / c_s, context vector for each title
                a = torch.cat((a, a2), 1)
                # a: (batch_size, 1000) / cat of c_g, c_s (c_g computed below)

            out = torch.cat((hx, a), 1)  # (batch_size, 1500) / [ h_t || c_t ] in paper
            outputs.append(out)
        l = torch.stack(outputs, 1)  # (batch_size, max abstract len, 1500)
        s = torch.sigmoid(self.switch(l))  # (batch_size, max abstract len, 1) / 1-p (p in equation 8 in paper)
        o = self.out(l)  # (batch_size, max abstract len, target vocab size)
        o = torch.softmax(o, 2)
        o = s * o
        # compute copy attn
        _, z = self.mattn(l, (ents, entlens))
        # z: (batch_size, max_abstract_len, max_entity_num) / entities attended on each abstract word
        z = (1 - s) * z
        o = torch.cat((o, z), 2)
        o = o + (1e-6 * torch.ones_like(o))  # add epsilon to avoid underflow
        return o.log(), z, planlogits

    def maskFromList(self, size, l):
        '''
        :param size: (batch_size, title_len, 500)
        :param l: tensor of (batch_size,) / lengths of each example
        :return: (batch_size, 1, title_len) / 각 row의 길이 + 1보다 더 큰 부분만 1, else 0
        '''
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
            tencs, _ = self.tenc(b.src)
            tmask = self.maskFromList(tencs.size(), b.src[1]).unsqueeze(1)
        ents = b.ent
        entlens = ents[2]
        ents = self.le(ents)
        if self.graph:
            gents, glob, grels = self.ge(b.rel[0], b.rel[1], (ents, entlens))
            hx = glob
            # hx = ents.max(dim=1)[0]
            keys, mask = grels
            mask = mask == 0
        else:
            mask = self.maskFromList(ents.size(), entlens)
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
        a = self.attn(hx.unsqueeze(1), keys, mask=mask).squeeze(1)
        if self.args.title:
            a2 = self.attn2(hx.unsqueeze(1), tencs, mask=tmask).squeeze(1)
            a = torch.cat((a, a2), 1)
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
            a = self.attn(hx.unsqueeze(1), keys, mask=mask).squeeze(1)
            if self.args.title:
                a2 = self.attn2(hx.unsqueeze(1), tencs, mask=tmask).squeeze(1)
                # a =  a + (self.mix(hx)*a2)
                a = torch.cat((a, a2), 1)
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
                    tencs = tencs.repeat(len(beam.beam), 1, 1)
                    tmask = tmask.repeat(len(beam.beam), 1, 1)
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
                    tencs = tencs[:len(beam.beam)]
                    tmask = tmask[:len(beam.beam)]
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
