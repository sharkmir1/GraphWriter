import os
from collections import Counter
from collections.abc import Iterable
from copy import copy
import bisect

import numpy as np
import torch
from torchtext import data

import pargs as arg

import ipdb

### TODO: Make Dataloader class

class Dataset(object):
    def __init__(self, args, data_dir, eval_path, train_path=None):
        self.args = args

        self.INP = data.Field(sequential=True, batch_first=True, init_token="<start>", eos_token="<eos>", include_lengths=True)  # Title
        self.OUTP = data.Field(sequential=True, batch_first=True, init_token="<start>", eos_token="<eos>", include_lengths=True)  # Gold Abstract, preprocessed
        self.TGT = data.Field(sequential=True, batch_first=True, init_token="<start>", eos_token="<eos>")
        self.NERD = data.Field(sequential=True, batch_first=True, eos_token="<eos>")  # Entity Type (ner data)
        self.ENT = data.RawField()  # Entity
        self.REL = data.RawField()  # Relation between entities
        self.SORDER = data.RawField()
        self.SORDER.is_target = False
        self.REL.is_target = False
        self.ENT.is_target = False
        self.fields = [("src", self.INP), ("ent", self.ENT), ("nerd", self.NERD), ("rel", self.REL), ("out", self.OUTP), ("sorder", self.SORDER)]

        if train_path is None:
            testset = data.TabularDataset(path=os.path.join(data_dir, eval_path), format='tsv', fields=self.fields)
            # TODO: test time 때 vocab 재활용하도록
            self.test_iter = self._make_iters(testset, batch_size=args.vbsz, is_eval=True)
        else:
            trainset, validset = data.TabularDataset.splits(path=data_dir, train=train_path, validation=eval_path, format='tsv', fields=self.fields)
            self._make_vocabs(trainset)
            self.train_iters = self._make_iters(trainset, split_by_len=True, batch_size=args.bsz)
            self.valid_iter = self._make_iters(validset, batch_size=args.bsz[-1], is_eval=True)

        # print("\nVocab Sizes:")
        # for x in self.fields:
        #     try:
        #         print(x[0], len(x[1].vocab))
        #     except:
        #         try:
        #             print(x[0], len(x[1].itos))
        #         except:
        #             pass


    def _make_graphs(self, r, n_ent):
        '''
        :param r: relation string of a dataset row
        :param ent: number of entities in a dataset row
        :return: adj, rel matrices
        '''
        # convert triples to entlist with adj and rel matrices
        triples = r.strip().split(';')
        triples = [[int(y) for y in triple.strip().split()] for triple in triples]
        rel = [2]  # global root node (i.e. 'ROOT')
        adjsz = n_ent + 1 + (2 * len(triples))
        adj = torch.zeros(adjsz, adjsz)  # adjacency matrix (# Entity + Global root + # relations) ^2
        
        for i in range(n_ent):
            # connect with ROOT (idx: n_ent)
            adj[i, n_ent] = 1
            adj[n_ent, i] = 1
        for i in range(adjsz):
            adj[i, i] = 1
        for trp in triples:
            # append REL node and its inverse REL_INV node
            rel.extend([trp[1] + 3, trp[1] + 3 + self.REL.size])
            a = trp[0]
            b = trp[2]
            c = n_ent + len(rel) - 2 # idx of this REL
            d = n_ent + len(rel) - 1 # idx of this REL_INV
            adj[a, c] = 1
            adj[c, b] = 1
            adj[b, d] = 1
            adj[d, a] = 1

        rel = torch.LongTensor(rel)  # rel: ['ROOT', 'REL 1', 'REL 1_inv', 'REL 2', 'REL 2_inv', ...]
        return (adj, rel)

    def _make_vocabs(self, dataset: data.TabularDataset):        
        print('Building Vocab... ')

        # Output Vocab
        # generics => indices are at the last of the vocab
        # also includes indexed generics, (e.g. <method_0>)
        # len(vocab) = 11738
        self.OUTP.build_vocab(dataset, min_freq=self.args.outunk)
        generics = ['<method>', '<material>', '<otherscientificterm>', '<metric>', '<task>']  # Entity Types
        self.OUTP.vocab.itos.extend(generics)
        for x in generics:
            self.OUTP.vocab.stoi[x] = self.OUTP.vocab.itos.index(x)

        # Target Vocab
        # Same as Output Vocab, except for the indexed generics' indices
        # len(vocab) = 11738 / <method_0>, <material_0> ... : 11738, <method_1>, ... : 11739 and so on.
        self.TGT.vocab = copy(self.OUTP.vocab)
        offset = len(self.TGT.vocab.itos)
        for t in [tag[1:-1] for tag in generics]:
            for n in range(40):
                s = "<{}_{}>".format(t, n)
                self.TGT.vocab.stoi[s] = offset + n

        # Entity Type Vocab
        # Indices for not-indexed generics are same with those of output vocab
        self.NERD.build_vocab(dataset, min_freq=0)
        for x in generics:
            self.NERD.vocab.stoi[x] = self.OUTP.vocab.stoi[x]

        # Title Vocab
        self.INP.build_vocab(dataset, min_freq=self.args.entunk)

        # Relation Vocab
        # Adds relations.vocab + inverse of relations.vocab
        self.REL.special = ['<pad>', '<unk>', 'ROOT']
        with open(os.path.join(self.args.datadir, self.args.relvocab)) as f:
            rvocab = [x.strip() for x in f.readlines()]
            self.REL.size = len(rvocab)
            rvocab += [x + "_INV" for x in rvocab]
            rvocab = self.REL.special + rvocab
        self.REL.itos = rvocab

        # Entity Vocab (Words in entry phrases)
        words = Counter()
        with open(os.path.join(self.args.datadir, self.args.trainfile), encoding='utf-8') as f:
            for l in f:
                ents = ' '.join(l.split("\t")[1].split(" ; "))
                tokenized = ents.split()
                # TODO: support other tokenizer than word? 
                words.update(tokenized)

        # NOTE: pad를 원래 1로 놓았었는데 제정신인가
        self.ENT.itos = ['<pad>', '<unk>'] + list(words.keys())
        # self.ENT.itos = sorted(list(set(ents.split(" "))))
        self.ENT.stoi = {x: i for i, x in enumerate(self.ENT.itos)}

        print('Finished making vocab')

    def batchify(self, b):
        # b.fields: src, ent, nerd, rel, out, sorder, tgt, rawent, sordertgt
        ents, phrase_lens = zip(*b.ent)  # b.ent: (batch_size,) / list of x.ent
        # ents: (batch_size,) / tuple of (num of entity, longest entity len) per each dataset row
        # phrase_lens: (batch_size,) / tuple of (num of entity) per each dataset row i.e. 각 row의 각 entity의 단어 개수
        
        ents, ent_lens = self._batchify_ent(ents)
        # ents: (batch-sum of num of entities in each row, longest entity len)
        # ent_lens: (batch_size,) / tensor of number of entities in each row
        ents = ents.to(self.args.device)

        adj, rel = zip(*b.rel)  # b.rel: (batch_size,) / list of x.rel
        # adj: (batch_size,) / tuple of adjacency matrices per each dataset row
        # rel: (batch_size,) / tuple of list of relations per each dataset row
        if self.args.sparse:
            b.rel = [adj, self._list_to_device(rel)]
        else:
            b.rel = [self._list_to_device(adj), self._list_to_device(rel)]  # [[adj1, adj2, ...], [rel1, rel2, ...]]
        
        if self.args.plan:
            b.sordertgt = self._list_to_device(self._pad_list(b.sordertgt))
        
        phrase_lens = torch.cat(phrase_lens, 0).to(self.args.device)
        # phrase_lens: (sum of num of entities in each row, ) / tensor of 각 entity의 단어 개수
        ent_lens = ent_lens.to(self.args.device)
        b.ent = (ents, phrase_lens, ent_lens)
        return b

    def _batchify_ent(self, ents):
        # ents[i]: i번째 샘플(row)의 entity들을 나타내는 텐서, 각 텐서는 (entity 개수, max entity word-length)
        lens = [x.size(0) for x in ents]  # lens: list of # of entities in each dataset row
        m = max([x.size(1) for x in ents])  # m: longest entity len in the batch
        data = [self._pad(x.transpose(0, 1), m).transpose(0, 1) for x in ents]
        data = torch.cat(data, 0)
        return data, torch.LongTensor(lens)  # data: pads each x in adj to (,m), and concat them

    def _vectorize_ents(self, ex, field):
        # returns tensor and lens
        ex = [[field.stoi[x] if x in field.stoi else 0 for x in y.strip().split(" ")] for y in ex.split(";")]
        return self._pad_list(ex)

    def _split_bsz_by_len(self, dataset, cutoffs=[0, 100, 220]):
        splits = [[] for _ in cutoffs]
        for x in dataset:
            belong_idx = bisect.bisect_right(cutoffs, len(x.out)) - 1
            splits[belong_idx].append(x)
        return splits

    def _make_iters(self, dataset, batch_size, split_by_len=False, is_eval=False):
        if split_by_len:
            splits = self._split_bsz_by_len(dataset)
            datasets = [data.Dataset(split, self.fields) for split in splits]
        else:
            datasets = [dataset]
        
        iters = []
        for i, dataset in enumerate(datasets):
            # print(len(dataset.examples), end=' ')
            
            for x in dataset:
                x.rawent = x.ent.split(" ; ")
                # (entity-phrase data tensor, lengths of each phrase)
                x.ent = self._vectorize_ents(x.ent, self.ENT) # (num of entity, longest entity len), (num of entity)
                # (adj matrix, rel node tensor)
                x.rel = self._make_graphs(x.rel, len(x.ent[1]))
                if self.args.sparse:
                    sp = [row.nonzero().squeeze(1) for row in x.rel[0]]
                    x.rel = (sp, x.rel[1])
                x.tgt = x.out
                # removes tag indices for out (e.g. <method_0> => <method>)
                x.out = [w.split("_")[0] + ">" if "_" in w else w for w in x.out]
                x.sordertgt = torch.LongTensor([int(x) + 3 for x in x.sorder.split(" ")])
                x.sorder = [[int(y) for y in x.strip().split(" ")] for x in x.sorder.split("-1")[:-1]]

            dataset.fields["tgt"] = self.TGT
            dataset.fields["rawent"] = data.RawField()
            dataset.fields["sordertgt"] = data.RawField()
            dataset.fields["rawent"].is_target = False
            dataset.fields["sordertgt"].is_target = False

            bsz = batch_size[i] if isinstance(batch_size, Iterable) else batch_size
            # if train, shuffled and not sorted
            # if eval, sorted and not shuffled
            iters.append(data.Iterator(dataset, bsz, device=self.args.device, sort_key=lambda x: len(x.out), repeat=False, train=not is_eval))
        
        return iters

    def reverse(self, x, ents):
        ents = ents[0]
        vocab = self.TGT.vocab
        s = ' '.join(
            [vocab.itos[y] if y < len(vocab.itos) else ents[y - len(vocab.itos)].upper() for j, y in enumerate(x)])
        if "<eos>" in s: s = s.split("<eos>")[0]
        return s

    # NOTE: 원래 1로 채웠음
    def _pad_list(self, l, ent=0):
        lens = [len(x) for x in l]
        m = max(lens)
        return torch.stack([self._pad(torch.tensor(x), m, ent) for x in l], 0), torch.LongTensor(lens)

    def _pad(self, tensor, length, ent=0):
        return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).fill_(ent)])

    def _list_to_device(self, l):
        return [x.to(self.args.device) for x in l]
