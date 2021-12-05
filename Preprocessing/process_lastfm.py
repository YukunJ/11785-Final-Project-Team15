import codecs
from datetime import datetime
import os
import time


items_freq=dict()
items_id=dict()
itemcount=1
start_time = time.time()
print("Begin Processing Lastfm data: ... (it is going to take a long while guys!")
f_len = 19150868
print("Begin first pass through 'userid-timestamp-artid-artname-traid-traname.tsv' file ...")

with codecs.open('userid-timestamp-artid-artname-traid-traname.tsv', encoding='utf-8') as f:
    count = 0
    for line in f:
        count += 1
        if count % 2000000 == 0:
            print("Progress: {}/{} = {:.3f}%".format(count, f_len, 100 * count / f_len))
        lines = line.strip('\n').strip('\r').split('\t')
        if len(lines) < 5:
            continue
        userid = lines[0]
        time = datetime.strptime(lines[1], "%Y-%m-%dT%H:%M:%SZ")
        itemid = lines[2]
        if len(itemid) < 2:
            continue
        if itemid not in items_id:
            items_id[itemid]=str(itemcount)
            items_freq[itemid]=1
            itemcount+=1
        else:
            items_freq[itemid] += 1
print("End first pass through 'userid-timestamp-artid-artname-traid-traname.tsv' file")
if len(items_freq)>40000:
    items_freq=dict(sorted(items_freq.items(), key=lambda d:d[1],reverse=True)[:40000])

print("Begin writing into 'items.artist.txt' files ... ")
fff = codecs.open('items.artist.txt', encoding='utf-8', mode='w')
for k in items_freq:
    fff.write(items_id[k]+os.linesep)
fff.close()
print("Finish writing into 'items.artist.txt' files")

last_time=None
last_user=None
session=list()
ff = codecs.open('all.artist.txt', encoding='utf-8', mode='w')
print("Begin second pass through 'userid-timestamp-artid-artname-traid-traname.tsv' file ...")
with codecs.open('userid-timestamp-artid-artname-traid-traname.tsv', encoding='utf-8') as f:
    count = 0
    for line in f:
        count += 1
        if count % 2000000 == 0:
            print("Progress: {}/{} = {:.3f}%".format(count, f_len, 100 * count / f_len))
        lines = line.strip('\n').strip('\r').split('\t')
        if len(lines) < 5:
            continue
        userid = lines[0]
        time = datetime.strptime(lines[1], "%Y-%m-%dT%H:%M:%SZ")
        itemid = lines[2]
        if len(itemid) < 2 or itemid not in items_freq:
            continue

        itemid=items_id[itemid]

        if last_time is None:
            session.append(itemid)
            last_user=userid
            last_time=time
            continue

        if (last_time-time).total_seconds()>28800 or last_user!=userid:
            if 50>len(session)>1 :
                ff.write(', '.join(list(reversed(session))) + os.linesep)
                ff.flush()
            session = list()
            last_user = userid
            last_time = time
        else:
            if len(session)==0 or itemid !=session[-1]:
                session.append(itemid)
            last_user = userid
            last_time = time
ff.close()
print("Finish second pass through 'userid-timestamp-artid-artname-traid-traname.tsv' file")
print("All Done!!!")

end_time = time.time()
