import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence


class SeqEncoder(nn.Module):
    def __init__(self, config):
        super(SeqEncoder, self).__init__()

        self.emb_dim = config.gru_emb
        self.hid_dim = config.gru_hid
        self.out_dim = config.gru_hid
        self.num_layer = config.num_layers
        self.max_len = config.max_len
        self.dropout = config.drop_ratio

        # here `146` is the length of tran_dict _a2i
        self.emb = torch.nn.Embedding(200, self.emb_dim)
        # self.emb = torch.nn.Embedding(get_vac_size() + 10, self.emb_dim)

        self.lin1 = nn.Linear(self.emb_dim, self.hid_dim)

        # remind: the num_layer will increase the risk of optimizer becoming NAN
        self.rnn = nn.GRU(
            input_size=self.hid_dim,
            num_layers=self.num_layer,
            hidden_size=self.hid_dim,
            bidirectional=True,
            batch_first=True,
        )
        self.return_mlp = nn.Linear(self.hid_dim * 2, self.out_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.hid_dim * self.max_len * 2, self.hid_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hid_dim, self.out_dim),
        )


    def forward(self, batch, return_node=False):
        x, length = batch.seq_feat, batch.seq_len

        batch_size = length.shape[0]
        x = x.view(batch_size, -1)
        embed_x = self.lin1(self.emb(x))
        # print("embed size {}".format(embed_x.shape))

        cur_lens = length.data
        cur_lens = cur_lens.to("cpu")
        packed_embed_x = pack_padded_sequence(
            embed_x,
            batch_first=True,
            lengths=cur_lens,
            #   lengths=length,
            enforce_sorted=False,
        )

        packed_embed_x, _ = self.rnn(packed_embed_x)
        x, token_length = pad_packed_sequence(
            packed_embed_x, batch_first=True, total_length=self.max_len
        )

        if return_node:
            x = self.return_mlp(x)
            return x, token_length
        else:
            input = torch.gather(x, dim=1, index=(cur_lens-1).to(x.device).unsqueeze(1).repeat(1, x.shape[-1]).unsqueeze(1))
            x = self.return_mlp(input)
            x = x.view(batch_size, -1)
            return x, None


class SeqModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.seqModel = SeqEncoder(args)
        self.classifier = nn.Sequential(nn.Linear(args.gru_hid, args.gru_hid),
                                        nn.LeakyReLU(),
                                        nn.Linear(args.gru_hid, args.num_tasks))

    def forward(self, data):
        seq_repr, _ = self.seqModel(data, return_node=False)
        output = self.classifier(seq_repr)
        return output


