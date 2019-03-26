import csv
import sys

s = set()
selected = []
with open(sys.argv[1], 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if row[4] == 'Hip-Hop':
            selected.append(row)
	    s.add(row[3])

print(sorted(list(s)))

print(len(selected))

tot = 0
for row in selected:
    lyrics = row[-1]
    x = lyrics.split(" ")
    charcount = sum([len(x) for x in x])
    tot += charcount

print(tot)

f = open(sys.argv[2], 'w')
for row in selected:
    lyrics = row[-1]
    f.write(lyrics)
  
f.close()
