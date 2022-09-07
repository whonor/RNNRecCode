from config import args
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RNN_layers(nn.Module):
    def __init__(self, args, inputs_hidden):
        super(RNN_layers, self).__init__()

        self.rnn = nn.RNN(input_size=inputs_hidden,  # nn.GRU() nn.LSTM()
                          hidden_size=args.hidden_size,
                          num_layers=args.rnn_layers,
                          batch_first=True)
        self.args = args

    def forward(self, seq, seqLength):
        _, order_index = torch.sort(seqLength, dim=0, descending=True)
        _, unorder_index = torch.sort(order_index, dim=0)
        order_seq = seq.index_select(0, order_index).to(device)
        order_seqLength = seqLength.index_select(0, order_index).to(device)
        pack_seq = pack_padded_sequence(order_seq, order_seqLength, batch_first=True).to(device)  #
        out, _rnn = self.rnn(pack_seq)  #
        pad_seq, rnn_len = pad_packed_sequence(out, batch_first=True)  #
        rnn_len = rnn_len.to(device)
        pad_seq = pad_seq.index_select(0, unorder_index)
        rnn_len = rnn_len.index_select(0, unorder_index)
        rnn_len = (rnn_len - 1).view(rnn_len.size(0), 1, -1)
        rnn_len = rnn_len.repeat(1, 1, self.args.hidden_size)
        out_hidden = torch.gather(pad_seq, 1, rnn_len)
        return out_hidden


class Model(nn.Module):
    def __init__(self, args, deviceEngEmbedding=None, deviceNumEmbedding=None, eventEmbedding=None, descEmbedding=None):
        super(Model, self).__init__()
        if deviceEngEmbedding is None:
            self.deviceEng_embedding = nn.Embedding(100, 150).to(device)
            self.deviceNum_embedding = nn.Embedding(100, 150).to(device)
            self.event_embedding = nn.Embedding(100, 300).to(device)
            self.desc_embedding = nn.Embedding(100, 300).to(device)
        else:
            self.deviceEng_embedding = nn.Embedding.from_pretrained(deviceEngEmbedding, freeze=False).to(device)
            self.deviceNum_embedding = nn.Embedding.from_pretrained(deviceNumEmbedding, freeze=False).to(device)
            self.event_embedding = nn.Embedding.from_pretrained(eventEmbedding, freeze=False).to(device)
            self.desc_embedding = nn.Embedding.from_pretrained(descEmbedding, freeze=False).to(device)
        self.device_rnn_layer = RNN_layers(args, inputs_hidden=600).to(device)
        self.event_rnn_layer = RNN_layers(args, inputs_hidden=args.event_hidden).to(device)
        self.MLP = nn.Sequential(
            nn.Linear(args.hidden_size * 2 + args.desc_hidden_size, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, 1),
            nn.Sigmoid()
        ).to(device)

    def forward(self, deviceEng, deviceNum, eventID, deviceLength, eventLength, description):
        deviceEng_embed = self.deviceEng_embedding(deviceEng)
        deviceNum_embed = self.deviceNum_embedding(deviceNum)

        device_embed = torch.cat([deviceEng_embed, deviceNum_embed], dim=-1)
        event_embed = self.event_embedding(eventID).to(device)

        device_rnn = self.device_rnn_layer(device_embed, deviceLength).squeeze(1)
        event_rnn = self.event_rnn_layer(event_embed, eventLength).squeeze(1)

        desc_embed = self.desc_embedding(description)
        desc_embed_max, _ = torch.max(desc_embed, dim=1)

        concat_all = torch.cat([device_rnn, event_rnn, desc_embed_max], dim=1).view(deviceEng.size(0),
                                                                                    -1)  # batch, 1+1+len(desc), hidden

        output = self.MLP(concat_all)
        output = output.view(output.size(0), -1).squeeze(-1)
        return output
