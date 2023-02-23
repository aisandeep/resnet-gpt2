import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from dataset import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
from test import encoderImage, structure_image_beam_search, cell_image_beam_search, visualize_att

print('STARTING EVAL')

data_folder = "output"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
checkpoint = "/home/dev.narayanan/pubtabep/checkpoint_table.pth.tar"
word_map_structure_file = "/home/dev.narayanan/pubtabep/PUB_TAB_EXP/output/WORDMAP_STRUCTURE.json"
word_map_cell_file = "/home/dev.narayanan/pubtabep/PUB_TAB_EXP/output/WORDMAP_CELL.json"
# Load model
checkpoint = torch.load(checkpoint)
decoder_structure = checkpoint['decoder_structure']
decoder_cell = checkpoint["decoder_cell"]
decoder_structure = decoder_structure.to(device)
decoder_cell.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()

print('middle')

# Load word map (word2ix)
with open(word_map_structure_file, 'r') as j:
    word_map_structure = json.load(j)
with open(word_map_cell_file, "r") as j:
    word_map_cell = json.load(j)

id2word_stucture = id_to_word(word_map_structure)
id2word_cell = id_to_word(word_map_cell)

vocab_size_structure = len(word_map_structure)
vocab_size_cell = len(word_map_cell)

# Normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def evaluation(image_path):
    print('encoder starting')
    encoder_out =   encoderImage(encoder, image_path)
    print('encoder stopped')

    seq, alphas, hidden_states = structure_image_beam_search(
        encoder_out, decoder_structure, word_map_structure, beam_size=vocab_size_structure)
    print('beam search started')
    cells = []
    html = ""
    for index, s in seq:
        html += id2word_stucture[str(s)]
        if id2word_stucture[str(s)] == "<td>" or id2word_stucture[str(s)] == ">":
            hidden_state_structure = hidden_states[index]
            seq_cell, alphas = cell_image_beam_search(
                encoder_out, decoder_cell, word_map_cell, hidden_state_structure, beam_size=vocab_size_cell)

            html_cell = convertId2wordSentence(id2word_cell, seq_cell)
            html += html_cell

    print(html)

    
if __name__ == '__main__':
    print('starting the model')
    evaluation('/home/dev.narayanan/pubtabep/PUB_TAB_EXP/pubtabnet/train/PMC1626454_002_00.png')