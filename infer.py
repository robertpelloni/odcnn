from model import *
from model_crnn import CRNNOnsetMaxNet
from music_processor import safe_load_pickle
import pickle
import os

if __name__ == '__main__':
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    arch = 'cnn'
    if '--arch' in sys.argv:
        arch_idx = sys.argv.index('--arch')
        if arch_idx + 1 < len(sys.argv):
            arch = sys.argv[arch_idx + 1].lower()

    if arch == 'crnn':
        net = CRNNOnsetMaxNet()
    else:
        net = convNet()
    net = net.to(device)

    song = safe_load_pickle('./data/pickles/test_data.pickle')

    target_type = None
    for arg in sys.argv[1:]:
        if arg in ('don', 'ka'):
            target_type = arg
            break
    if target_type is None:
        target_type = 'don'

    if target_type == 'don':
        model_file = './models/don_crnn.pth' if arch == 'crnn' and os.path.exists('./models/don_crnn.pth') else './models/don_model.pth'
        map_loc = None if torch.cuda.is_available() else 'cpu'
        net.load_state_dict(torch.load(model_file, map_location=map_loc))

        inference = net.infer(song.feats, device, minibatch=512)
        inference = np.reshape(inference, (-1))

        with open('./data/pickles/don_inference.pickle', mode='wb') as f:
            pickle.dump(inference, f)

    elif target_type == 'ka':
        model_file = './models/ka_crnn.pth' if arch == 'crnn' and os.path.exists('./models/ka_crnn.pth') else './models/ka_model.pth'
        map_loc = None if torch.cuda.is_available() else 'cpu'
        net.load_state_dict(torch.load(model_file, map_location=map_loc))

        inference = net.infer(song.feats, device, minibatch=512)
        inference = np.reshape(inference, (-1))

        with open('./data/pickles/ka_inference.pickle', mode='wb') as f:
            pickle.dump(inference, f)
