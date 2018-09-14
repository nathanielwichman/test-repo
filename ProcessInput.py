import string
import random

class ProcessInput:
	
	def getBigQuestion(test_percent, cv_percent, size, replace_uncommon=False):
		data, max_length = ProcessInput.processBigQuestion(size, replace_uncommon)
		data = ProcessInput.getTranslations(data)
		new_data, test, cv = ProcessInput.getTestSets(data, test_percent, cv_percent)
		random.shuffle(new_data)	
		return data, new_data, test, cv, max_length

	def processBigQuestion(size, replace_uncommon=False):
		questions = open("bigquestion.txt")
		processed = list()
		add = False
		translator = str.maketrans('', '', string.punctuation) #strips punctuation
		max_length = 0

		total_sentences = 0

		test_dict = dict() # for finding uncommon words

		for line in questions:
			if total_sentences > size:
				break

			if add:
				total_sentences += 1
				line = line.strip().lower()
				line = line.translate(translator)
				
				line = line.split(" ")
				processed.append(line)
				add = False

				# Processing, maybe turn off at runtime
				max_length = max(max_length, len(line))
				for word in line:
					if word in test_dict:
						test_dict[word] += 1
					else:
						test_dict[word] = 1
			
			elif line[0] == "Q": # Only add questions to test set
				add = True
		
		words = [0, 0, 1, 0]
		print (words)
		for word in test_dict:
			count = test_dict[word]
			if count == 1:
				words[0] += 1
			elif count == 2:
				words[1] += 1
			elif count == 3:
				words[2] += 1
			else:
				words[3] += 1
		print (words)

		return processed, max_length

	def getTranslations(data):
		paired_data = list()
		for sentence in data:
			paired_data.append((sentence, list(reversed(sentence))))
			#for more data
			#paired_data.append((list(reversed(sentence)), sentence))
		return paired_data

	def getTestSets(data, percent_test, percent_cv):
		new_data = data.copy() # Since elements are removed

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

