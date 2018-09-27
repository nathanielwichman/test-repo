# -*- coding: utf-8 -*-

import string
import random
from ProcessInput import ProcessInput


_, data, _, _, _ = ProcessInput.getChnEng(ProcessInput(), 10, 10, 200)
print(data)




"""
eline = 0
eblank = 0

cline = 0
cblank = 0

for line in e:
	eline += 1
	if line.strip() == "":
		print(eline)
		break
		eblank += 1

#for line in c:
#	cline += 1
#	if line.strip() == "":
#		cblank += 1
	
print(eline)
print(eblank)
print("")
print(cline)
print(cblank)

alpha = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQR`STUVWX’YZ -,,.“”';:[]()-$%#*&		"
num = "0123456789"
c = 0
#f = open("casict2015_en.txt")
#z = open("casict2015_ch.txt")
f = open("short_eng.txt")
z = open("short_parsed.txt")
#f_out = open("short_eng.txt", "w")
#z_out = open("short_chn.txt", "w")

print("files open")
check_num = 20
c2 = 0
b = 0
b2 = 0

for line in f:
	c += 1
	if c % 1000 == 0:
		print(line)

for line in z:
	c2 += 1
	if c2 % 1000 == 0:
		print(line)

for line in f:
	c += 1
	if c < 300000:
		f_out.write(line)
	else:
		break

for line in z:
	c2 += 1
	if c2 < 300000:
		z_out.write(line)
	else:
		break




for line in f:
	out.write(line)
	line = line.strip()
	c += 1
	if len(line) == 0:
		continue
	bad_count = 0
	to_check = min(len(line), check_num)
	for i in range(to_check):
		if line[i] not in alpha and line[i] not in num:
			bad_count += 1
	if bad_count > 2:		
		o_c += 1
		#print(str(c) + ": " + line)
		
print()
print(c)

f = open("finished.en")

lines = 0
vocab = dict()
# < 4, 4-12, 13-20, 20-40, 40+
length = [0, 0, 0, 0, 0]
#translator = str.maketrans('', '', string.punctuation)
examples = []
ending = set()

for line in f:
	lines += 1
	#split = line
	#processed = line.strip().lower()
	processed = line.strip()
	split = processed.split(" ")
	ending.add(split[-1])
	for word in split:
		if word in vocab:
			vocab[word] += 1
		else:
			vocab[word] = 1

	wl = len(split)
	if wl < 4:
		length[0] += 1
	elif wl < 13:
		length[1] += 1
	elif wl < 21:
		length[2] += 1
	elif wl < 40:
		length[3] += 1
	else:
		length[4] += 1

#         1, 2, 3, 4-10, 11+
counts = [0, 0, 0, 0, 0]
for _, count in vocab.items():
	if count == 1:
		counts[0] += 1
	elif count == 2:
		counts[1] += 1
	elif count == 3:
		counts[2] += 1
	elif count < 11:
		counts[3] += 1
	else:
		counts[4] += 1
	
print("total lines = " + str(lines) + "\r\n")
print("sentence length dist: " + str(length) + "\r\n")
print("word occurance dist:  " + str(counts) + "\r\n")
print("examples: \r\n")
#print(ending, file=w)
print(examples)
"""
#c = 0
#for word, count in vocab.items():
#	if c > 100:
#		break
#	elif count == 1:
#		print(word + "   ", file=w)
#		c += 1

