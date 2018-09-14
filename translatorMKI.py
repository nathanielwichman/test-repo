import torch

from ProcessInput import ProcessInput
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

START = 0
END = 1

class LSTMEncoder(nn.Module):
	def __init__(self, embedding_dim, hidden_dim, input_vocab_size):
		super(LSTMEncoder, self).__init__()
		self.hidden_dim = hidden_dim
		self.word_embeddings = nn.Embedding(input_vocab_size, embedding_dim)

		# Get word meanings
		self.lstm = nn.LSTM(embedding_dim, hidden_dim)
		
		self.hidden = self.init_hidden()

	def init_hidden(self):
		return (torch.zeros(1, 1, self.hidden_dim), 
    		    torch.zeros(1, 1, self.hidden_dim))

	def forward(self, sentence):
		embeds = self.word_embeddings(sentence)
		lstm_out, self.hidden = self.lstm(embeds.view(1, 1, -1), self.hidden)
		return lstm_out

class LSTMDecoder(nn.Module):
	def __init__(self, embedding_dim, hidden_dim, output_size):
		super(LSTMDecoder, self).__init__()
		self.hidden_dim = hidden_dim

		self.embedding = nn.Embedding(output_size, hidden_dim)
		self.lstm = nn.LSTM(hidden_dim, hidden_dim)
		self.out = nn.Linear(hidden_dim, output_size)
		self.softmax = nn.LogSoftmax(dim=1)

		self.hidden = self.init_hidden()

	def forward(self, input_values):
		input_values = self.embedding(input_values).view(1, 1, -1)
		non_linear = F.relu(input_values)
		output, self.hidden = self.lstm(input_values, self.hidden)
		return self.softmax(self.out(output[0]))


	def init_hidden(self):
		return (torch.zeros(1, 1, self.hidden_dim),
				torch.zeros(1, 1, self.hidden_dim))

def generateToy(test=False):
	sentences = [
		"food is good",
		"money is good",
		"walking is good",
		"famine is good",
		"gold is good",
		"life is good",
		"he walked to the bar",
		"the two men talked quietly",
		"falling is bad for you",
		"nobody wants bad food"
	]
	if test:
		sentences.append("good is falling")
		sentences.append("zebras like to eat food")
	return [(s.split(" "), (list(reversed(s.split(" "))))) for s in sentences]

def tag(data):
	tag_dict = {"START_TOKEN": START, "END_TOKEN": END}

	for sentence, _ in data:
		for word in sentence:
			if word not in tag_dict:
				tag_dict[word] = len(tag_dict)
	return tag_dict

def process(data, sentence, translation):
	idx = []
	idt = []
	for w in sentence:
		idx.append(data[w])

	for w in translation:
		idt.append(data[w])

	idx.append(END)
	idt.append(END)

	return torch.tensor(idx, dtype=torch.long), torch.tensor(idt, dtype=torch.long)

def train_step(encoder, decoder, sentence, translation, max_length, encoder_optimizer, decoder_optimizer, loss_func):
	encoder.zero_grad()
	decoder.zero_grad()
	
	encoder.hidden = encoder.init_hidden()
	
	#encoder_outputs = torch.zeros(max_length, encoder.hidden_size)

	for i in range(sentence.size()[0]):
		_ = encoder(sentence[i])
		#encoder_outputs[i] = output[0, 0]
	
	decoder.hidden = encoder.hidden
	decoder_input = torch.tensor([[START]])

	#translation_attempt = []
	loss = 0

	for i in range(len(translation)):
		output = decoder(decoder_input)
		results, index = torch.topk(output, 1)
		decoder_input = index.squeeze().detach()
		
		#translation_attempt.append(index.item())
		
		loss += loss_func(output, translation[i].unsqueeze(0))
		if decoder_input.item() == END:
			break
	#print ("sentence: " + str(sentence))
	#print(translation_attempt)
	loss.backward()
	encoder_optimizer.step()
	decoder_optimizer.step()

	return loss.item() / len(sentence)

def test(encoder, decoder, sentence, translation, loss_func, max_length):
	with torch.no_grad():
		encoder.hidden = encoder.init_hidden()
		
		for i in range(sentence.size()[0]):
			_ = encoder(sentence[i])
		
		decoder.hidden = encoder.hidden
		decoder_input = torch.LongTensor([[START]])
		
		loss = 0
		decoded_words = list()

		for i in range(max_length):
			output = decoder(decoder_input)
			results, index = torch.topk(output, 1)
			#print(translation[i])
			#print(index)
			if i < len(translation):
				loss += loss_func(output, translation[i].unsqueeze(0))
			if index.item() == END:
				 decoded_words.append(END)
				 break
			else:
				decoded_words.append(index.item())
			decoder_input = index.squeeze().detach()
		return decoded_words, (loss.item() / len(decoded_words))

def toWords(sentence, data):
	toPrint = ""
	for w in sentence:
		toPrint += data[w]
		toPrint += " "
	return toPrint

def training(data, test_tensors, cv_tensors, vocab_size, epochs, dicti):
	reverse_data = dict((v, k) for k, v in dicti.items())
	encoder = LSTMEncoder(10, 300, vocab_size)
	decoder = LSTMDecoder(10, 300, vocab_size)

	encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.003)
	decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.003)

	loss_function = nn.NLLLoss()
		
	print("starting training")
	for epoch in range(epochs):
		epoch_loss = 0
		step = 0
		for sentence, translation in data:
			# Progress report
			if step % 100 == 0:
				print("on iteration " + str(step))

			step_loss = train_step(encoder, decoder, sentence, translation, 40, 
				encoder_optimizer, decoder_optimizer, loss_function)
			epoch_loss += step_loss
			step += 1

		print("EPOCH " + str(epoch) + ": " + str(step_loss / len(data)))
	
	print("test results:")
	test_loss = 0
	for sentence, translation in test_tensors:
		attempt, loss = test(encoder, decoder, sentence, translation, loss_function, 40)
		test_loss += loss
		print("sentence: " + toWords(sentence.numpy(), reverse_data))
		print("translat: " + toWords(attempt, reverse_data))
	print("training loss: " + str(test_loss / len(test_tensors)))

	print("cv loss:")
	cv_loss = 0
	for sentence, translation in cv_tensors:
		attempt, loss = test(encoder, decoder, sentence, translation, loss_function, 40)
		cv_loss += loss
		print("sentence: " + toWords(sentence.numpy(), reverse_data))
		print("translat: " + toWords(attempt, reverse_data))
	print("training loss: " + str(cv_loss / len(cv_tensors)))


_, training_data, test_vals, cv_vals, max_length = ProcessInput.getBigQuestion(10, 10, 10000)
data = tag(training_data + cv_vals)
print (len(training_data))
tensor_data = []
for sentence, translation in training_data:
	tensor_data.append(process(data, sentence, translation))
test_tensors = []
for sentence, translation in test_vals:
	test_tensors.append(process(data, sentence, translation))
cv_tensors = []
for sentence, translation in cv_vals:
	cv_tensors.append(process(data, sentence, translation))
training(tensor_data, test_tensors, cv_tensors, len(data), 120, data)




