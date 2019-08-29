import sys
from random import shuffle
import os
from math import exp
import torch
from torch import nn
from torch.nn import functional as F
from dataset import Dataset
from pargs import pargs, dynArgs
from models.newmodel import Model


def update_lr(o, args, epoch):
    if epoch % args.lrstep == 0:
        o.param_groups[0]['lr'] = args.lrhigh
    else:
        o.param_groups[0]['lr'] -= args.lrchange


def train(model, optimizer, dataset, args):
    """
    input words => indices all removed from tags / trained to generate indexed tags
    target words => indices included
    """
    loss = 0
    ex = 0
    # NOTE: 셔플링 필요한가?
    # trainorder = [('t1', dataset.t1_iter), ('t2', dataset.t2_iter), ('t3', dataset.t3_iter)]
    # shuffle(trainorder)
    for train_iter in dataset.train_iters:
        for count, batch in enumerate(train_iter):
            if count % 100 == 99:
                print(ex, "of like 40k -- current avg loss ", (loss / ex))
            batch = dataset.batchify(batch)

            model.zero_grad()
            pred, dist_copy, plan_logits = model(batch)  # p: (batch_size, max abstract len, target vocab size + max entity num)

            tgt = batch.tgt[:, 1:].contiguous().view(-1).to(args.device)  # exclude first word from each target
            loss = F.nll_loss(pred[:,:-1,:].contiguous().view(-1, pred.size(2)), tgt, ignore_index=1) 
            
            # copy coverage (each element at least once)
            if args.cl:
                dist_copy = dist_copy.max(1)[0]
                coverage_loss = nn.functional.mse_loss(dist_copy, torch.ones_like(dist_copy))
                loss = loss + args.cl * coverage_loss
            if args.plan:
                plan_loss = nn.functional.cross_entropy(plan_logits.view(-1, plan_logits.size(2)), batch.sordertgt[0].view(-1), ignore_index=1)
                loss = loss + args.plweight * plan_loss

            loss.backward()
            loss += loss.item() * len(batch.tgt)

            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            ex += len(batch.tgt)
    loss = loss / ex
    print("AVG TRAIN LOSS: ", loss, end="\t")
    if loss < 100: print(" PPL: ", exp(loss))


def evaluate(model, dataset, args):
    print("Evaluating", end="\t")
    model.eval()
    accumulated_loss = 0
    ex = 0
    for batch in dataset.val_iter:
        batch = dataset.batchify(batch)
        pred, *_ = model(batch)
        pred = pred[:, :-1, :]

        tgt = batch.tgt[:, 1:].contiguous().view(-1).to(args.device)
        loss = F.nll_loss(pred.contiguous().view(-1, pred.size(2)), tgt, ignore_index=1)
        if ex == 0:
            gen = pred[0].max(1)[1]
            print(dataset.reverse(gen, batch.rawent))
        accumulated_loss += loss.item() * len(batch.tgt)
        ex += len(batch.tgt)

    valid_loss = accumulated_loss / ex
    print("VAL LOSS: ", valid_loss, end="\t")
    if loss < 100: print(" PPL: ", exp(valid_loss))
    model.train()
    return valid_loss


def main(args):
    try:
        os.stat(args.save)
        input("Save File Exists, OverWrite? <CTL-C> for no")
    except:
        os.mkdir(args.save)

    dataset = Dataset(args, data_dir=args.datadir, eval_path=args.validfile, train_path=args.trainfile)
    args = dynArgs(args, dataset)
    model = Model(args)
    print(args.device)
    model = model.to(args.device)
    if args.ckpt:
        #
        # with open(args.save+"/commandLineArgs.txt") as f:
        #   clargs = f.read().strip().split("\n")
        #   argdif =[x for x in sys.argv[1:] if x not in clargs]
        #   assert(len(argdif)==2);
        #   assert([x for x in argdif if x[0]=='-']==['-ckpt'])
        #
        cpt = torch.load(args.ckpt)
        model.load_state_dict(cpt)
        start_epoch = int(args.ckpt.split("/")[-1].split(".")[0]) + 1
        args.lr = float(args.ckpt.split("-")[-1])
        print('ckpt restored')

    else:
        with open(args.save + "/commandLineArgs.txt", 'w') as f:
            f.write("\n".join(sys.argv[1:]))
        start_epoch = 0
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # early stopping based on Val Loss
    prev_loss = 1000000

    for epoch_i in range(start_epoch, args.epochs):
        print("epoch ", epoch_i, "lr", optimizer.param_groups[0]['lr'])
        train(model, optimizer, dataset, args)
        vloss = evaluate(model, dataset, args)
        if args.lrwarm:
            update_lr(optimizer, args, epoch_i)
        print("Saving model")
        torch.save(model.state_dict(),
                   args.save + "/" + str(epoch_i) + ".vloss-" + str(vloss)[:8] + ".lr-" + str(optimizer.param_groups[0]['lr']))
        if vloss > prev_loss:
            if args.lrdecay:
                print("decay lr")
                optimizer.param_groups[0]['lr'] *= 0.5
        prev_loss = vloss


if __name__ == "__main__":
    args = pargs()
    main(args)
