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

base_output_path = './output/RepeatNet/'
base_data_path = './datasets/demo/'
dir_path = os.path.dirname(os.path.realpath(__file__))
epoches=60
embedding_size=100
hidden_size=100
#item_vocab_size=10+1
item_vocab_size = 43097+1+1

def train(args):
    batch_size = 256

    #train_dataset = RepeatNetDataset(base_data_path+'demo.train')
    train_dataset = RepeatNetDataset(base_data_path+'digi_train.txt')
    train_size=train_dataset.len

    model = RepeatNet(embedding_size, hidden_size, item_vocab_size)
    init_params(model)

    trainer = CumulativeTrainer(model, None, None, None, 1)
    model_optimizer = optim.Adam(model.parameters())
    #model_scheduler = optim.lr_scheduler.StepLR(model_optimizer, step_size=2, gamma=0.8)
    print("Begin training...")
    for i in range(epoches):
        if i > 0 and i % 3 == 0:
            print("Shrinking optimizer LR by half at epoch {}".format(i))
            for g in model_optimizer.param_groups:
                g['lr'] *= 0.8
        print("start epoch : {}/{}".format(i+1, epoches))
        trainer.train_epoch('train', train_dataset, collate_fn, batch_size, i, model_optimizer)
        trainer.serialize(i, output_path=base_output_path)
        print("end epoch : {}/{}".format(i+1, epoches))
        torch.save(trainer.model.state_dict(), base_output_path+"lastest_repeatnet.pt")

def infer(args):
    batch_size = 256
    epoches = 1
    valid_dataset = RepeatNetDataset(base_data_path + 'digi_valid.txt')
    test_dataset = RepeatNetDataset(base_data_path + 'digi_test.txt')

    for i in range(epoches):
        print('epoch', i)
        file = base_output_path+"repeatnet.pt"

        if os.path.exists(file):
            model = RepeatNet(embedding_size, hidden_size, item_vocab_size)
            model.load_state_dict(torch.load(file, map_location='cpu'))
            print("successfully load model state dict.")
            trainer = CumulativeTrainer(model, None, None, None, 1)
            print("start get into trainer.predict() functionality.")
            print("doing validation...")
            rs = trainer.predict('infer', valid_dataset, collate_fn, batch_size, i, base_output_path, 'valid')
            print("doing testing...")
            rs = trainer.predict('infer', valid_dataset, collate_fn, batch_size, i, base_output_path, 'test')
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
