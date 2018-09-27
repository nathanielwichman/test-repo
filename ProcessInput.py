import string
import random

class ProcessInput:
    UNCOMMON_WORD_TAG = "<U>"
    ENG_TAG = "<E>"
    NUMBER_TAG = "<N>"

    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    numbers = "0123456789"

    def getChnEng(self, test_percent, cv_percent, size, replace_uncommon=False):
        data, max_length = ProcessInput.processEng(self, size, replace_uncommon)
        data = ProcessInput.processChn(self, data)
        new_data, test, cv = ProcessInput.getTestSets(self, data, test_percent, cv_percent)
        random.shuffle(new_data)
        return data, new_data, test, cv, max_length

    def processChn(self, data):
        chinese_data = open("finished.zh")
        paired_data = []
        count = 0
        for line in chinese_data:
            if count >= len(data):
                break
            paired_data.append((line.strip().split(), data[count]))
            count += 1

        return paired_data


    def processEng(self, size, replace_uncommon=False):
        english = open("finished.en")
        data = []
        test_dict = dict()
        max_length = 0
        count = 0

        for line in english:
            count += 1
            if count > size:
                break
            processed = line.strip().split()
            for word in processed:
                if word not in test_dict:
                    test_dict[word] = 1
                else:
                    test_dict[word] += 1
            max_length = max(max_length, len(processed))
            data.append(processed)

        if replace_uncommon:
            to_replace = []
            for word, count in test_dict.items():
                if count < 2:
                    to_replace.append(word)
            for sentence in data:
                for i in range(len(sentence)):
                    if sentence[i] in to_replace:
                        sentence[i] = "<U>"
        return data, max_length

    def getBigQuestion(self, test_percent, cv_percent, size, replace_uncommon=False):
        data, max_length = ProcessInput.processBigQuestion(self, size, replace_uncommon)
        data = ProcessInput.getTranslations(self, data)
        new_data, test, cv = ProcessInput.getTestSets(self, data, test_percent, cv_percent)
        random.shuffle(new_data)
        return data, new_data, test, cv, max_length

    def processBigQuestion(self, size, replace_uncommon=False):
        questions = open("bigquestion.txt")
        processed = list()
        add = False
        # Python3 only?
        translator = str.maketrans('', '', string.punctuation) #strips punctuation
        total_sentences = 0

        for line in questions:
            if total_sentences > size:
                break

            if add:
                total_sentences += 1
                line = line.strip().lower()
                line = line.translate(translator)
                #line = line.translate(None, string.punctuation)

                # line = line.split(" ")
                processed.append(line)
                add = False

            elif line[0] == "Q": # Only add questions to test set
                add = True

        processed = set(processed)
        finished = []
        max_length = 0

        test_dict = dict() # for finding uncommon words
        for sentence in processed:
            split_sentence = sentence.split(" ")
            finished.append(split_sentence)

            max_length = max(max_length, len(split_sentence))

            # Processing, maybe turn off at runtime
            for word in split_sentence:
                if word in test_dict:
                    test_dict[word] += 1
                else:
                    test_dict[word] = 1

        # Replaces uncommon words with shared token
        # Note that this will re-introduce duplicate sentences

        if replace_uncommon:
            to_replace = []
            for word, count in test_dict.items():
                if count < 2:
                    to_replace.append(word)
            for sentence in finished:
                for i in range(len(sentence)):
                    if sentence[i] in to_replace:
                        sentence[i] = "<U>"

        # Probably remove for large data sets, expensive with little payoff
        actually_remove = []
        #for i in range(len(finished)):
        #   for j in range(len(finished)):
        #       if i != j and finished[i] == finished[j]:
        #           actually_remove.append(finished[i])
        #           break

        #pre_l = len(finished)
        #for sentence in actually_remove:
        #    finished.remove(sentence)
        #print("removed: " + str(pre_l - len(finished)))

        for i in range(min(20, (size - 1))):
            print(finished[i])

        return finished, max_length

    def getTranslations(self, data):
        paired_data = list()
        for sentence in data:
            paired_data.append((sentence, list(reversed(sentence))))
            #for more data
            paired_data.append((list(reversed(sentence)), sentence))
        return paired_data

    def getTestSets(self, data, percent_test, percent_cv):
        new_data = list(data) # Since elements are removed

        # Get cv set, delete test examples from data
        number_cv = int(round((percent_cv / 100)* len(new_data)))

        cv_index = random.sample(range(0, len(new_data)), number_cv)
        cv_set = [new_data[i] for i in cv_index]
        for i in sorted(cv_index, reverse=True):
            del new_data[i]

        # Get training test set
        number_test = int(round((percent_test / 100) * len(new_data)))

        test_index = random.sample(range(0, len(new_data)), number_test)
        test_set = [new_data[i] for i in test_index]

        return new_data, test_set, cv_set

