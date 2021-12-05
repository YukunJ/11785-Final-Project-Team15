import sys
sys.path.append('./')
from RepeatNet.Dataset import *
from torch import optim
from Common.CumulativeTrainer import *
import torch.backends.cudnn as cudnn
import argparse
from RepeatNet.Model import *
import codecs
import numpy as np
import random
import time

def get_ms():
    return time.time() * 1000

def init_seed(seed=None):
    if seed is None:
        seed = int(get_ms() // 1000)

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def evaluate(result_path):
    recall_10 = recall_20 = 0
    mrr_10 = mrr_20 = 0
    total_line = 0
    with open(result_path, mode='r', encoding='utf-8') as f_handle:
        f = f_handle.readlines()
    for line in f:
        total_line += 1
        ranking, label = line.strip().split('|')
        label = int(label[1:-1])
        ranking = ranking.strip('[]')
        ranking = list(map(int, ranking.split(',')))
        for i in range(0, 20):
            # rank is i+1
            if ranking[i] == label:
                if i+1 <= 10:
                    recall_10 += 1
                    mrr_10 += 1 / (i+1)
                recall_20 += 1
                mrr_20 += 1 / (i+1)
    recall_10 /= total_line
    recall_20 /= total_line
    mrr_10 /= total_line
    mrr_20 /= total_line
    print("Result for file: {}".format(result_path))
    print("Recall@10 : {:.4f}".format(100*recall_10))
    print("Recall@20 : {:.4f}".format(100*recall_20))
    print("MRR@10 : {:.4f}".format(100*mrr_10))
    print("MRR@20 : {:.4f}".format(100*mrr_20))
    print("#=---------------------------=#")
    
base_output_path = './output/RepeatNet/'
base_data_path = './datasets/demo/'
dir_path = os.path.dirname(os.path.realpath(__file__))
epoches = 15
embedding_size=100
hidden_size=100
item_vocab_size = 7731+1+1

"""
Yukun editing: æ·»åŠ category side infoçš„size
"""
categ_side_size = 996+1+1 

def train(args):
    batch_size = 128

    train_dataset = RepeatNetDataset(base_data_path+'movie_train.txt', base_data_path+'movie_train.txt')
    train_size=train_dataset.len

    model = RepeatNet(embedding_size, hidden_size, item_vocab_size)
    init_params(model)

    trainer = CumulativeTrainer(model, None, None, None, 1)
    model_optimizer = optim.Adam(model.parameters(), weight_decay=1e-6)
    #model_scheduler = optim.lr_scheduler.StepLR(model_optimizer, step_size=2, gamma=0.8)
    print("Begin training...")
    for i in range(1, epoches+1):
        if i > 0 and i % 5 == 0:
            print("Shrinking optimizer LR by half at epoch {}".format(i))
            for g in model_optimizer.param_groups:
                g['lr'] *= 0.5
        print("start epoch : {}/{}".format(i, epoches))
        trainer.train_epoch('train', train_dataset, collate_fn, batch_size, i, model_optimizer)
        trainer.serialize(i, output_path=base_output_path)
        print("end epoch : {}/{}".format(i, epoches))
        torch.save(trainer.model.state_dict(), base_output_path+"lastest_repeatnet_" + str(i) + ".pt")
        torch.save(trainer.model.state_dict(), base_output_path+"lastest_repeatnet" + ".pt")
        print("Sleeping 10 secs for <model storage> in drive buffer refresh...")
        time.sleep(10)
        print("Testing on this epoch:")
        infer(args)
        print("Sleeping 10 secs for <txt file> in drive buffer refresh...")
        time.sleep(10)
        evaluate(base_output_path+"test_result.txt")
        

def infer(args):
    batch_size = 128
    epoches = 1
    test_dataset = RepeatNetDataset(base_data_path + 'movie_test.txt', base_data_path + 'movie_test.txt')
    
    for i in range(1, epoches+1):
        file = base_output_path+"lastest_repeatnet.pt"

        if os.path.exists(file):
            model = RepeatNet(embedding_size, hidden_size, item_vocab_size)
            device = torch.device("cuda")
            model.load_state_dict(torch.load(file, map_location=device))
            trainer = CumulativeTrainer(model, None, None, None, 1)
            print("start get into trainer.predict() functionality.")
            print("doing testing...")
            rs = trainer.predict('infer', test_dataset, collate_fn, batch_size, i, base_output_path, 'test')
            print("Finish storing, return")
            return
            #-------------------#
            file = codecs.open(base_output_path+'result/'+str(i)+'.'+str(args.local_rank)+'.valid', mode='w', encoding='utf-8')
            f = open(base_output_path+"valid_result.txt", mode='w', encoding='utf-8')
            for data, output in rs:
                scores, index=output
                label=data['item_tgt']
                for j in range(label.size(0)):
                    file.write('[' + ','.join([str(id) for id in index[j, :50].tolist()]) + ']|[' + ','.join([str(id) for id in label[j].tolist()]) + ']' + os.linesep)
                    f.write('[' + ','.join([str(id) for id in index[j, :50].tolist()]) + ']|[' + ','.join([str(id) for id in label[j].tolist()]) + ']' + os.linesep)
                    print("writing: ", '[' + ','.join([str(id) for id in index[j, :50].tolist()]) + ']|[' + ','.join([str(id) for id in label[j].tolist()]) + ']' + os.linesep)
            file.close()
            f.close()
            rs = trainer.predict('infer', test_dataset, collate_fn, batch_size, i, base_output_path)
            file = codecs.open(base_output_path + 'result/' + str(i)+'.'+str(args.local_rank)+'.test', mode='w', encoding='utf-8')
            f = open(base_output_path+"test_result.txt", mode='w', encoding='utf-8')
            for data, output in rs:
                scores, index = output
                label = data['item_tgt']
                for j in range(label.size(0)):
                    file.write('[' + ','.join([str(id) for id in index[j, :50].tolist()]) + ']|[' + ','.join([str(id) for id in label[j].tolist()]) + ']' + os.linesep)
                    f.write('[' + ','.join([str(id) for id in index[j, :50].tolist()]) + ']|[' + ','.join([str(id) for id in label[j].tolist()]) + ']' + os.linesep)
                    print("writing: ", '[' + ','.join([str(id) for id in index[j, :50].tolist()]) + ']|[' + ','.join([str(id) for id in label[j].tolist()]) + ']' + os.linesep)
            file.close()
            f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--mode", type=str)
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.distributed.init_process_group(backend='NCCL', init_method='env://')

    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True
    print(torch.__version__)
    print(torch.version.cuda)
    print(cudnn.version())
    init_seed(123456)

    if args.mode=='infer':
        infer(args)
    elif args.mode=='train':
        train(args)
