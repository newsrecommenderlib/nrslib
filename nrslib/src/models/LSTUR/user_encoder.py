import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import pytorch_lightning as pl


class UserEncoder(pl.LightningModule):
    def __init__(self, num_filters, long_short_term_method):
        super(UserEncoder, self).__init__()
        self.num_filters = num_filters
        self.long_short_term_method = long_short_term_method
        assert int(self.num_filters * 1.5) == self.num_filters * 1.5
        self.gru = nn.GRU(
            self.num_filters * 3,
            self.num_filters * 3 if self.long_short_term_method == 'ini'
            else int(self.num_filters * 1.5))

    def forward(self, user, clicked_news_length, clicked_news_vector):
        """
        Args:
            user:
                ini: batch_size, num_filters * 3
                con: batch_size, num_filters * 1.5
            clicked_news_length: batch_size,
            clicked_news_vector: batch_size, num_clicked_news_a_user, num_filters * 3
        Returns:
            (shape) batch_size, num_filters * 3
        """
        clicked_news_length[clicked_news_length == 0] = 1
        # 1, batch_size, num_filters * 3
        if self.long_short_term_method == 'ini':
            packed_clicked_news_vector = pack_padded_sequence(
                clicked_news_vector,
                clicked_news_length.cpu(),
                batch_first=True,
                enforce_sorted=False)
            _, last_hidden = self.gru(packed_clicked_news_vector,
                                      user.unsqueeze(dim=0))
            return last_hidden.squeeze(dim=0)
        else:
            packed_clicked_news_vector = pack_padded_sequence(
                clicked_news_vector,
                clicked_news_length,
                batch_first=True,
                enforce_sorted=False)
            _, last_hidden = self.gru(packed_clicked_news_vector)
            return torch.cat((last_hidden.squeeze(dim=0), user), dim=1)
