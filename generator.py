import torch
import argparse
from time import time
from dataset import Dataset
from models.newmodel import model
from pargs import pargs, dynArgs
from tqdm import tqdm


# import utils.eval as evalMetrics

def tgtreverse(tgts, entlist, order):
    entlist = entlist[0]
    order = [int(x) for x in order[0].split(" ")]
    tgts = tgts.split(" ")
    k = 0
    for i, x in enumerate(tgts):
        if x[0] == "<" and x[-1] == '>':
            tgts[i] = entlist[order[k]]
            k += 1
    return " ".join(tgts)


def test(args, dataset, model, epoch='cmdline'):
    args.vbsz = 1
    model = args.save.split("/")[-1]
    model.eval()
    k = 0
    data = dataset.mktestset(args)
    ofn = "../outputs/" + model + ".inputs.beam_predictions." + epoch
    pf = open(ofn, 'w')
    preds = []
    golds = []
    for b in tqdm(data):
        # if k == 10: break
        # print(k,len(data))
        b = dataset.batchify(b)
        '''
    p,z = m(b)
    p = p[0].max(1)[1]
    gen = dataset.reverse(p,b.rawent)
    '''
        gen = model.beam_generate(b, width=4, k=6, max_len=args.maxlen)
        gen.sort()  # sort 'done' sequences by their scores
        gen = dataset.reverse(gen.done[0].words, b.rawent)
        k += 1
        gold = dataset.reverse(b.tgt[0][1:], b.rawent)
        # print("GOLD\n"+gold)
        # print("\nGEN\n"+gen)
        # print()
        preds.append(gen.lower())
        golds.append(gold.lower())
        # tf.write(ent+'\n')
        pf.write(str(k) + '.\n')
        pf.write("GOLD\n" + gold.encode('ascii', 'ignore').decode('utf-8', 'ignore') + '\n')
        pf.write("GEN\n" + gen.encode('ascii', 'ignore').decode('utf-8', 'ignore').lower() + '\n\n')

    model.train()
    return preds, golds


'''
def metrics(preds,gold):
  cands = {'generated_description'+str(i):x.strip() for i,x in enumerate(preds)}
  refs = {'generated_description'+str(i):[x.strip()] for i,x in enumerate(gold)}
  x = evalMetrics.Evaluate()
  scores = x.evaluate(live=True, cand=cands, ref=refs)
  return scores
'''

if __name__ == "__main__":
    args = pargs()
    args.eval = True
    ds = Dataset(args, data_dir=args.datadir, eval_path=args.testfile)
    args = dynArgs(args, ds)
    m = model(args)
    cpt = torch.load(args.save)
    m.load_state_dict(cpt)
    m = m.to(args.device)
    m.args = args
    m.starttok = ds.OUTP.vocab.stoi['<start>']
    m.endtok = ds.OUTP.vocab.stoi['<eos>']
    m.eostok = ds.OUTP.vocab.stoi['.']
    args.vbsz = 1
    preds, gold = test(args, ds, m)
    '''
  scores = metrics(preds,gold)
  for k,v in scores.items():
    print(k+'\t'+str(scores[k]))
  '''
