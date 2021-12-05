import codecs
import os

train = codecs.open('lastfm_train.artist.txt', encoding='utf-8', mode='w')
session_max_len = 50
with codecs.open('train.artist.txt', encoding='utf-8') as f:
    for line in f:
        lines = line.strip('\n').strip('\r').split(', ')
        for i in range(len(lines)-1):
            train.write('['+', '.join(lines[0:i+1])+ ', ' + ', '.join(['0'] * (session_max_len-i-1)) + ']'+'|'+ '[' +lines[i+1]+']'+ os.linesep)

train.close()

dev = codecs.open('lastfm_valid.artist.txt', encoding='utf-8', mode='w')
with codecs.open('dev.artist.txt', encoding='utf-8') as f:
    for line in f:
        lines = line.strip('\n').strip('\r').split(', ')
        for i in range(len(lines)-1):
            dev.write('['+', '.join(lines[0:i+1])+ ', ' + ', '.join(['0'] * (session_max_len-i-1)) + ']'+'|'+ '[' +lines[i+1]+']'+ os.linesep)

dev.close()

test = codecs.open('lastfm_test.artist.txt', encoding='utf-8', mode='w')
with codecs.open('test.artist.txt', encoding='utf-8') as f:
    for line in f:
        lines = line.strip('\n').strip('\r').split(', ')
        for i in range(len(lines)-1):
            test.write('['+', '.join(lines[0:i+1])+ ', ' + ', '.join(['0'] * (session_max_len-i-1)) + ']'+'|'+ '[' +lines[i+1]+']'+ os.linesep)

test.close()
