
import torch

# 1. Tokenizer: split by whitespace
def word_tokenizer(sentence):
    "Tokenizes sentence into words (split on whitespace)"
    tokens = sentence.strip().split()
    return tokens


def encode_words(words, max_len, char_dict):
    """
    Converts tokenized sentence into list of token IDs 
    with <end> and padding
    """
    # Add <end> + padding until reaching max_len
    for i in range(max_len - len(words)):
        if i == 0:
            words.append('<end>')
        else:
            words.append('_')

    # Convert to IDs
    word_vec = [char_dict[w] for w in words]
    return word_vec


def decode_words(encoded_tensors, org_dict):
    "Decodes tensor of token IDs into sentence strings"
    sentences = []
    for i in range(encoded_tensors.shape[0]):
        encoded_tensor = encoded_tensors.cpu().numpy()[i, :]
        sentence = ''
        for j in range(encoded_tensor.shape[0]):
            idx = encoded_tensor[j]
            sentence += org_dict[idx] + ' '
        # Remove unwanted tokens
        sentence = sentence.replace('_ ', '')
        sentence = sentence.replace('<end> ', '')
        sentence = sentence.strip()  # remove trailing space
        sentences.append(sentence)
    return sentences



def vae_data_gen(data, char_dict, max_len=127):
    """
    Encodes input sentences to tensors with token ids

    Arguments:
        data (np.array, req): Array containing input sentences
        char_dict (dict, req): Dictionary mapping tokens to integer id
        max_len (int, opt): Maximum length for encoded sequences
    Returns:
        encoded_data (torch.tensor): Tensor containing encodings for each
                                     sentence
    """
    sentences = data
    del data
    sentences = [word_tokenizer(x) for x in sentences]
    encoded_data = torch.empty((len(sentences), max_len), dtype=torch.long)
    for j, sent in enumerate(sentences):
        encoded_sent = encode_words(sent, max_len-1, char_dict)
        encoded_sent = [char_dict["<start>"]] + encoded_sent   # prepend start token
        encoded_data[j,:] = torch.tensor(encoded_sent)
    return encoded_data



