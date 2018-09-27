import torch

import time
from ProcessInput import ProcessInput
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd.variable as Variable

torch.manual_seed(1)

START = 0
END = 1


class LSTMEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, input_vocab_size):
        super(LSTMEncoder, self).__init__()

        # embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(input_vocab_size, embedding_dim)

        # Get word meanings
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sentence, hidden):
        # embeds = self.word_embeddings(sentence)
        # for python 2
        embeds = self.word_embeddings(sentence)
        lstm_out, hidden = self.lstm(embeds.view(1, 1, -1), hidden)
        return lstm_out, hidden


class LSTMAttentionDecoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_size):
        super(LSTMAttentionDecoder, self).__init__()

        # embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(output_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.to_final = nn.Linear(hidden_dim * 2, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.LogSoftmax(2) # dim

    def score(self, ht, hs):
        # dot
        return torch.dot(ht.squeeze(), hs.squeeze()).item()

    def forward(self, input_values, hidden, input_hidden):
        # input_hidden = torch.transpose(input_hidden, 0, 2)

        input_values = self.embedding(input_values).view(1, 1, -1)
        non_linear = F.relu(input_values)
        _, hidden = self.lstm(non_linear, hidden)

        a = torch.ones(input_hidden.size(0))
        for i in range(input_hidden.size(0)):
            z = self.score(hidden[0], torch.index_select(input_hidden, 0, torch.tensor([i])))
            a[i] = z

        weighted_values = torch.mul(a.view(-1, 1, 1), input_hidden)
        c = torch.mean(weighted_values, 0)
        final_hidden = torch.tanh(self.to_final(torch.cat((c.view(1,1,-1), hidden[0]), 2)))
        mid = self.out(final_hidden)
        # p rint(mid)
        output = self.softmax(mid)
        # p rint(output)
        return output.squeeze(0), hidden, final_hidden  # for concat


class LSTMDecoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_size):
        super(LSTMDecoder, self).__init__()
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(output_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.LogSoftmax(1)  # dim

    def forward(self, input_values, hidden):
        input_values = self.embedding(input_values).view(1, 1, -1)
        non_linear = F.relu(input_values)
        output, hidden = self.lstm(non_linear, hidden)
        return self.softmax(self.out(output[0])), hidden

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))


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


def train_attention_step(encoder, decoder, sentence, translation, encoder_opt, decoder_opt, loss_func):
    encoder.zero_grad()
    decoder.zero_grad()

    hidden = encoder.init_hidden()

    _, hidden = encoder(sentence[0], hidden)
    hidden_states = torch.tensor(hidden[0])

    for i in range(1, len(sentence)):
        _, hidden = encoder(sentence[i], hidden)
        #print (hidden_states.size())
        hidden_states = torch.cat((hidden_states, hidden[0]), 0)

    decoder_input = torch.tensor([[START]])

    loss = 0

    for i in range(len(translation)):
        output, hidden, _ = decoder(decoder_input, hidden, hidden_states)
        results, index = torch.topk(output, 1)
        decoder_input = index.squeeze().detach()
        # print (output.size())
        # print(translation[i].unsqueeze(0).size())
        loss += loss_func(output, translation[i].unsqueeze(0))
        if decoder_input.item() == END:
            break

    loss.backward()
    encoder_opt.step()
    decoder_opt.step()
    return loss.item() / len(sentence)


def attention_test(encoder, decoder, sentence, translation, loss_func, mas_length):
    with torch.no_grad():
        hidden = encoder.init_hidden()

        hidden_states = torch.tensor(hidden[0])

        for i in range(sentence.size()[0]):
            _, hidden = encoder(sentence[i], hidden)
            hidden_states = torch.cat((hidden_states, hidden[0]), 0)

        decoder_input = torch.tensor([[START]])
        decoded_words = list()
        loss = 0

        for i in range(max_length):
            output, hidden, _ = decoder(decoder_input, hidden, hidden_states)
            results, index = torch.topk(output, 1)
            if i < len(translation):
                loss += loss_func(output, translation[i].unsqueeze(0))
            if index.item() == END:
                decoded_words.append(END)
                break
            else:
                decoded_words.append(index.item())
            decoder_input = index.squeeze().detach()

        return decoded_words, (loss.item() / len(decoded_words))


def train_step(encoder, decoder, sentence, translation, encoder_optimizer, decoder_optimizer, loss_func):
    encoder.zero_grad()
    decoder.zero_grad()

    hidden = encoder.init_hidden()

    for i in range(len(sentence)):  # sentence.size()[0]):
        _, hidden = encoder(sentence[i], hidden)

    decoder_input = torch.LongTensor([[START]])  # torch.tensor([[START]])

    # translation_attempt = []
    loss = 0

    for i in range(len(translation)):
        output, hidden = decoder(decoder_input, hidden)
        results, index = torch.topk(output, 1)
        decoder_input = index.squeeze().detach()
        print (output.size())
        print(translation[i].unsqueeze(0))
        # translation_attempt.append(index.item())

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
        hidden = encoder.init_hidden()

        for i in range(sentence.size()[0]):
            _, hidden = encoder(sentence[i], hidden)

        decoder_input = torch.LongTensor([[START]])

        loss = 0
        decoded_words = list()

        for i in range(max_length):
            output, hidden = decoder(decoder_input, hidden)
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

def output_data(attention, replace, training_size, vocab, epochs, lr, embed, hidden, test_size, cv_size, test_loss, cv_loss):
    FILE_NAME = "MKI_output.txt"
    writer = open(FILE_NAME, "a")
    writer.write(str(time.time()) + "\r\n")
    writer.write("Using attention?: " + str(attention) +"\r\n")
    writer.write("Replacing uncom?: " + str(replace) + "\r\n")
    writer.write("training size   : " + str(training_size) + "  vocab size: " + str(vocab) + "  epochs: " + str(epochs) + "\r\n")
    writer.write("learning rate   : " + str(lr) + "  word embed size: " + str(embed) + "  hidden size: " + str(hidden) + "\r\n")
    writer.write("test size       : " + str(test_size) + ",  test loss: " + str(test_loss) + "\r\n")
    writer.write("cv size         : " + str(cv_size) +   ",  cv loss  : " + str(cv_loss) + "\r\n")
    writer.write("\r\n")
    writer.close()
    print("results written to file: " + FILE_NAME)


def training(data, test_tensors, cv_tensors, vocab_size, epochs, dicti, max_length, alpha, replace, attention, embed_size, hidden_size):
    """
    :param data: list of tuples to train on(sentence, translation)
    :param test_tensors: list of tuples from training set to test on finished model
    :param cv_tensors: list of tuples not included in training set to test on finished model
    :param vocab_size: number of words in the vocab
    :param epochs: number of epochs to train for
    :param dicti: dictionary word: encoding (where encoding is just an index for a One hot unique vec)
    :param max_length: the longest sentence length (num words) that can be trained on or generated
    :param alpha: the learning rate for both the decoder and encoder
    :param replace: if true, uncommon words (only 1 occurrence) were replaced by a stand-in token
    :parma attention: if true, a decoder using global attention is used
    :param embed_size: word embeding vector size in both encoder and decoder
    :param hidden_size: length of hidden size in both encoder and decoder
    """

    reverse_data = dict((v, k) for k, v in dicti.items())

    encoder = LSTMEncoder(embed_size, hidden_size, vocab_size)

    if attention:
        decoder = LSTMAttentionDecoder(embed_size, hidden_size, vocab_size)
    else:
        decoder = LSTMAttentionDecoder(embed_size, hidden_size, vocab_size)

    encoder_optimizer = optim.Adam(encoder.parameters(), alpha) #lr
    decoder_optimizer = optim.Adam(decoder.parameters(), alpha) #lr

    loss_function = nn.NLLLoss()

    print("starting training")
    for epoch in range(epochs):
        epoch_loss = 0
        step = 0
        for sentence, translation in data:
            # Progress report
            if step % 100 == 0:
                print("on iteration " + str(step))
            step_loss = train_attention_step(encoder, decoder, sentence, translation, \
                                            encoder_optimizer, decoder_optimizer, loss_function)
            #step_loss = train_step(encoder, decoder, sentence, translation,
            #	encoder_optimizer, decoder_optimizer, loss_function)
            epoch_loss += step_loss
            step += 1

        print("EPOCH " + str(epoch) + ": " + str(step_loss / len(data)))

    output_name = "last_output.txt"
    print("output printed to: " + output_name)
    f =  open(output_name, "w")

    print("test results:", file=f)
    test_loss = 0
    for sentence, translation in test_tensors:
        attempt, loss = attention_test(encoder, decoder, sentence, translation, loss_function, max_length)
        test_loss += loss
        print("sentence: " + toWords(sentence.numpy(), reverse_data), file=f)
        print("translat: " + toWords(attempt, reverse_data), file=f)
    print("training loss: " + str(test_loss / len(test_tensors)), file=f)

    print("cv loss:", file=f)
    cv_loss = 0
    for sentence, translation in cv_tensors:
        attempt, loss = attention_test(encoder, decoder, sentence, translation, loss_function, max_length)
        cv_loss += loss
        print("sentence: " + toWords(sentence.numpy(), reverse_data), file=f)
        print("translat: " + toWords(attempt, reverse_data), file=f)
    print("training loss: " + str(cv_loss / len(cv_tensors)), file=f)

    report = (attention, replace, len(data), vocab_size, epochs, alpha, embed_size, hidden_size, len(test_tensors),
              len(cv_tensors), (test_loss / len(test_tensors)), (cv_loss / len(cv_tensors)))

    output_data(*report)



print("test")
TRAINING_EXAMPLES = 300    # number of examples to train on
EPOCHS =            10       # number of epochs to train for (currently no mini-batch, so all examples used)
PERCENT_TEST =      10       # percent of training examples to test on
PERCENT_CV =        10       # percent of examples to reserve for only testing
ALPHA =             .001     # learning rate for both encoder and decoder
REPLACE =           False    # replace uncommon words with one symbol
ATTENTION =         False    # use a decoder model using attention
EMBED_SIZE =        10      # size of the word embedding vector
HIDDEN_SIZE =       300      # size of the hidden unit (encoder and decoder)


_, training_data, test_vals, cv_vals, max_length = ProcessInput.getChnEng(ProcessInput(), PERCENT_TEST, PERCENT_CV, TRAINING_EXAMPLES, REPLACE)
data = tag(training_data + cv_vals)
print("size = " + str(len(training_data)))
print("vocab = " + str(len(data)))
print("Attention: " + str(ATTENTION))
print("Replace: " + str(REPLACE))
print("Epochs: " + str(EPOCHS))
tensor_data = []
for sentence, translation in training_data:
    tensor_data.append(process(data, sentence, translation))
test_tensors = []
for sentence, translation in test_vals:
    test_tensors.append(process(data, sentence, translation))
cv_tensors = []
for sentence, translation in cv_vals:
    cv_tensors.append(process(data, sentence, translation))
training(tensor_data, test_tensors, cv_tensors, len(data), EPOCHS, data, max_length, ALPHA, REPLACE, ATTENTION, EMBED_SIZE, HIDDEN_SIZE)




