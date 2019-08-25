import torch
from torch import nn
from torch.nn import functional as F

tt = torch.cuda if torch.cuda.is_available() else torch


class beam_obj():
    def __init__(self, initword, initscore, h, c, last):
        self.words = [initword]  # each obj contains one starting word (after init token)
        self.score = initscore
        self.h = h
        self.c = c
        self.last = last
        self.firstwords = [initword]
        self.prevent = 0
        self.isstart = False


class Beam():

    def __init__(self, words, scores, hs, cs, last, beamsz, k, vsz):
        """

        :param words: (k) / top k word indices
        :param scores: (k) / top k words scores
        :param hs: [hx (1, 500)] * beam_size
        :param cs: [cx (1, 500)] * beam_size
        :param last: [c_t (1,1000)] * beam size
        :param beamsz: default 4
        :param k: default 6
        :param vsz: output vocab size (without entities, but including tags-both indexed and not indexed)
        """
        self.beamsz = beamsz
        self.k = k
        self.beam = []  # list containing beam objects
        for i in range(beamsz):
            self.beam.append(beam_obj(words[i].item(), scores[i].item(), hs[i], cs[i], last[i]))
        self.done = []
        self.vocabsz = vsz

    def sort(self, norm=True):
        if len(self.done) < self.beamsz:
            self.done.extend(self.beam)
            self.done = self.done[:self.beamsz]
        if norm:
            self.done = sorted(self.done, key=lambda x: x.score / len(x.words), reverse=True)
        else:
            self.done = sorted(self.done, key=lambda x: x.score, reverse=True)

    def dup_obj(self, obj):
        new_obj = beam_obj(None, None, None, None, None)
        new_obj.words = [x for x in obj.words]
        new_obj.score = obj.score
        new_obj.prevent = obj.prevent
        new_obj.firstwords = [x for x in obj.firstwords]
        new_obj.isstart = obj.isstart
        return new_obj

    def getwords(self):
        '''
        :return: last word of each beam
        '''
        return tt.LongTensor([[x.words[-1]] for x in self.beam])

    def geth(self):
        return torch.cat([x.h for x in self.beam], dim=0)

    def getc(self):
        return torch.cat([x.c for x in self.beam], dim=0)

    def getlast(self):
        return torch.cat([x.last for x in self.beam], dim=0)

    def getscores(self):
        '''
        :return: (beam size, k) / each row's entries are same (current scores)
        '''
        return tt.FloatTensor([[x.score] for x in self.beam]).repeat(1, self.k)

    def getPrevEnt(self):
        return [x.prevent for x in self.beam]

    def getIsStart(self):
        return [(i, self.beam[i].firstwords) for i in range(len(self.beam)) if self.beam[i].isstart]

    def update(self, scores, words, hs, cs, lasts):
        """
        :param scores: (beam size, 1, k) / top k word scores for each beam
        :param words: (beam size, 1, k) / top k word indices for each beam
        :param hs: (beam size, 500)
        :param cs: (beam size, 500)
        :param lasts: (beam size, 1000)
        :return:
        """
        beam = self.beam
        scores = scores.squeeze()  # (beam size, k)
        words = words.squeeze()
        k = self.k
        scores = scores + self.getscores()  # adds up word scores for each beam
        scores, idx = scores.view(-1).topk(len(self.beam))
        # (beam size,), (beam size,) / top beam size scores among all beams * k candidates
        newbeam = []
        for i, x in enumerate(idx):
            x = x.item()
            r = x // k;
            c = x % k
            w = words.view(-1)[x].item()  # w: word index according to one of top k indices
            new_obj = self.dup_obj(beam[r])  # copy the original beam object
            if w == self.endtok:
                new_obj.score = scores[i]
                self.done.append(new_obj)  # if a beam is over, append it to done
            else:
                if new_obj.isstart:  # if sentence starts
                    new_obj.isstart = False
                    new_obj.firstwords.append(w)
                if w >= self.vocabsz:  # if word in [<method_0>, ...]
                    new_obj.prevent = w
                if w == self.eostok:  # if sentence ends
                    new_obj.isstart = True
                new_obj.words.append(w)
                new_obj.score = scores[i]
                new_obj.h = hs[r, :].unsqueeze(0)
                new_obj.c = cs[r, :].unsqueeze(0)
                new_obj.last = lasts[r, :].unsqueeze(0)
                newbeam.append(new_obj)
        self.beam = newbeam
        return newbeam != []  # return false if all beams ended
