from collections import defaultdict, Counter
import numpy as np

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

if __name__ == '__main__':
    print("#=---------------------------=#")
    evaluate("test_result.txt")
