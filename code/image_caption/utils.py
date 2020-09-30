import os
import numpy as np
import h5py
import json
import torch
from scipy.misc import imread, imresize
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
import jsonlines

width_image = 512
height_image = 512


def create_input_files(dataset, json_file_path, image_folder="examples/", output_folder="ouput",
                       max_len_token_structure=300,
                       max_len_token_cell=100, width_image=512,
                       height_image=512):
    """
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param json_file_path: path of Json data with splits, structure token, cell token, img_path
    :param image_folder: folder with downloaded images
    :param output_folder: folder to save files
    :param max_len_token_structure: don't sample captions_structure longer than this length
    :param max_len_token_cell: don't sample captions_structure longer than this length
    """

    # Read Karpathy JSON
    with jsonlines.open('examples/PubTabNet_Examples.jsonl', 'r') as reader:
        imgs = list(reader)

    # Read image paths and captions for each image
    train_image_captions_structure = []
    train_image_captions_cells = []
    train_image_paths = []

    valid_image_captions_structure = []
    valid_image_captions_cells = []
    valid_image_paths = []

    test_image_captions_structure = []
    test_image_captions_cells = []
    test_image_paths = []
    word_freq_structure = Counter()
    word_freq_cells = Counter()

    for img in imgs:
        word_freq_structure.update(img["html"]["structure"]["tokens"])

        for cell in img["html"]["cells"]:
            word_freq_cells.update(cell["tokens"])

        captions_structure = []
        caption_cells = []
        path = os.path.join(image_folder, img['filename'])
        if len(img["html"]["structure"]["tokens"]) <= max_len_token_structure:
            captions_structure.append(img["html"]["structure"]['tokens'])
            for cell in img["html"]["cells"]:
                caption_cells.append(cell["tokens"])

            if img["split"] == "train":
                train_image_captions_structure.append(captions_structure)
                train_image_captions_cells.append(caption_cells)
                train_image_paths.append(path)
            elif img["split"] == "val":
                valid_image_captions_structure.append(captions_structure)
                valid_image_captions_cells.append(caption_cells)
                valid_image_paths.append(path)
            else:
                test_image_captions_structure.append(captions_structure)
                test_image_captions_cells.append(caption_cells)
                test_image_paths.append(path)

    # create vocabluary structure
    words_structure = [w for w in word_freq_structure.keys()]
    word_map_structure = {k: v + 1 for v, k in enumerate(words_structure)}
    word_map_structure['<unk>'] = len(word_map_structure) + 1
    word_map_structure['<start>'] = len(word_map_structure) + 1
    word_map_structure['<end>'] = len(word_map_structure) + 1
    word_map_structure['<pad>'] = 0

    # create vocabluary cells
    words_cell = [w for w in word_freq_cells.keys()]
    word_map_cell = {k: v + 1 for v, k in enumerate(words_cell)}
    word_map_cell['<unk>'] = len(word_map_cell) + 1
    word_map_cell['<start>'] = len(word_map_cell) + 1
    word_map_cell['<end>'] = len(word_map_cell) + 1
    word_map_cell['<pad>'] = 0

    # save vocabluary to json
    with open(os.path.join(output_folder, 'WORDMAP_' + "STRUCTURE" + '.json'), 'w') as j:
        json.dump(word_map_structure, j)

    with open(os.path.join(output_folder, 'WORDMAP_' + "CELL" + '.json'), 'w') as j:
        json.dump(word_map_cell, j)

    # store image and encoding caption to h5 file
    for impaths, imcaps_structure, imcaps_cell, split in [(train_image_paths, train_image_captions_structure, train_image_captions_cells, 'train'),
                                                          (valid_image_paths, valid_image_captions_structure,
                                                           valid_image_captions_cells, 'val'),
                                                          (test_image_paths, test_image_captions_structure, test_image_captions_cells, 'test')]:
        with h5py.File(os.path.join(output_folder, split + '_IMAGES_.hdf5'), 'a') as h:
            images = h.create_dataset(
                'images', (len(impaths), 3, width_image, height_image), dtype='uint8')
            print("\nReading %s images and captions, storing to file...\n" % split)
            enc_captions_structure = []
            enc_captions_cells = []
            cap_structure_len = []
            cap_cell_len = []
            for i, path in enumerate(tqdm(impaths)):
                captions_structure = imcaps_structure[i]
                captions_cell = imcaps_cell[i]
                img = imread(impaths[i])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = imresize(
                    img, (width_image, height_image), interp="cubic")
                img = img.transpose(2, 0, 1)
                # Save image to HDF5 file
                images[i] = img

                # encode caption cell and structure
                for j, c in enumerate(captions_structure):
                    enc_c = [word_map_structure['<start>']] + [word_map_structure.get(word, word_map_structure['<unk>']) for word in c] + [
                        word_map_structure['<end>']] + [word_map_structure['<pad>']] * (max_len_token_structure - len(c))
                    c_len = len(c) + 2
                    enc_captions_structure.append(enc_c)
                    cap_structure_len.append(c_len)
                for j, c in enumerate(captions_cell):
                    enc_c = [word_map_cell['<start>']] + [word_map_cell.get(word, word_map_cell['<unk>']) for word in c] + [
                        word_map_cell['<end>']] + [word_map_cell['<pad>']] * (max_len_token_cell - len(c))
                    c_len = len(c) + 2
                    enc_captions_cells.append(enc_c)
                    cap_cell_len.append(c_len)
            with open(os.path.join(output_folder, split + '_CAPTIONS_STRUCTURE' + '.json'), 'w') as j:
                json.dump(enc_captions_structure, j)
            with open(os.path.join(output_folder, split + '_CAPLENS_STRUCTURE' + '.json'), 'w') as j:
                json.dump(cap_structure_len, j)
            with open(os.path.join(output_folder, split + '_CAPTIONS_CELL' + '.json'), 'w') as j:
                json.dump(enc_captions_cells, j)
            with open(os.path.join(output_folder, split + '_CAPLENS_CELL' + '.json'), 'w') as j:
                json.dump(cap_cell_len, j)


def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.

    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.

    :param emb_file: file containing embeddings (stored in GloVe format)
    :param word_map: word map
    :return: embeddings in the same order as the words in the word map, dimension of embeddings
    """

    # Find embedding dimension
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    # Create tensor to hold embeddings, initialize
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(
            lambda n: n and not n.isspace(), line[1:])))

        # Ignore word if not in train_vocab
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    bleu4, is_best):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'checkpoint_' + data_name + '.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)
