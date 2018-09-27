import re

UNCOMMON_WORD_TAG = "<U>"
ENG_TAG = "<E>"
NUMBER_TAG = "<N>"

letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
numbers = "0123456789"
punc = "()\"“”\'[]{}"
end_punc = ".,;:!?"

output_file = "finished.en"
in_file = "short_eng.txt"

def parseEnglish(line):
    new_line = []
    for word in line:
        new_word = []
        for i in range(len(word)):
            letter = word[i]
            if letter in punc or (letter in end_punc and (i == len(word) - 1)):
                if len(new_word) > 0:
                    new_line.append("".join(new_word))
                new_line.append(letter)
                new_word = []
            else:
                new_word.append(letter)
        if len(new_word) > 0:
            new_line.append("".join(new_word))
    return new_line

def parseOddChinese(line):
    new_word = []
    for word in line:
        if word[0] in letters:
            new_word.append(ENG_TAG)
            if word[-1] == "." or word[-1] == "?" or word[-1] == "!":
                new_word.append("。")
        elif word[0] in letters:
            if [i in numbers for i in word]:
                new_word.append(NUMBER_TAG)
        else:
            new_word.append(word)
    return new_word


r = open(in_file)
w = open(output_file, "w")
#c = 500

for line in r:
    split = line.strip().split(" ")
    result = parseEnglish(split)
    #print(result)
    #print("a" + result[-1] + "a")
    for i in range(len(result) - 1):
        w.write(result[i])
        w.write(" ")
    w.write(result[-1])
    w.write("\n")

    #c -= 1
    #if c <= 0:
    #   break

