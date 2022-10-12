import torch
import torch.nn as nn
import torch.nn.functional as F
from ..components.attention.additive import AdditiveAttention

class NewsEncoder(torch.nn.Module):
    def __init__(self, num_words,word_embedding_dim,num_categories,
                 num_filters,window_size,query_vector_dim,dropout_probability, pretrained_word_embedding):
        super(NewsEncoder, self).__init__()
        self.num_words=num_words
        self.word_embedding_dim=word_embedding_dim
        self.num_categories=num_categories
        self.num_filters=num_filters
        self.window_size=window_size
        self.query_vector_dim=query_vector_dim
        self.dropout_probability=dropout_probability
        self.pretrained_word_embedding=pretrained_word_embedding
        if self.pretrained_word_embedding is None:
            self.word_embedding = nn.Embedding(self.num_words,
                                               self.word_embedding_dim,
                                               padding_idx=0)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(
                self.pretrained_word_embedding, freeze=False, padding_idx=0)
        self.category_embedding = nn.Embedding(self.num_categories,
                                               self.num_filters,
                                               padding_idx=0)
        assert self.window_size >= 1 and self.window_size % 2 == 1
        self.title_CNN = nn.Conv2d(
            1,
            self.num_filters,
            (self.window_size, self.word_embedding_dim),
            padding=(int((self.window_size - 1) / 2), 0))
        self.title_attention = AdditiveAttention(self.query_vector_dim,
                                                 self.num_filters)

    def forward(self, news):
        """
        Args:
            news:
                {
                    "category": batch_size,
                    "subcategory": batch_size,
                    "title": batch_size * num_words_title
                }
        Returns:
            (shape) batch_size, num_filters * 3
        """
        # Part 1: calculate category_vector

        # batch_size, num_filters
        category_vector = self.category_embedding(news['category'])

        # Part 2: calculate subcategory_vector

        # batch_size, num_filters
        subcategory_vector = self.category_embedding(
            news['subcategory'])

        # Part 3: calculate weighted_title_vector

        # batch_size, num_words_title, word_embedding_dim
        title_vector = F.dropout(self.word_embedding(news['title']),
                                 p=self.dropout_probability,
                                 training=self.training)
        # batch_size, num_filters, num_words_title
        convoluted_title_vector = self.title_CNN(
            title_vector.unsqueeze(dim=1)).squeeze(dim=3)
        # batch_size, num_filters, num_words_title
        activated_title_vector = F.dropout(F.relu(convoluted_title_vector),
                                           p=self.dropout_probability,
                                           training=self.training)
        # batch_size, num_filters
        weighted_title_vector = self.title_attention(
            activated_title_vector.transpose(1, 2))

        # batch_size, num_filters * 3
        news_vector = torch.cat(
            [category_vector, subcategory_vector, weighted_title_vector],
            dim=1)
        return news_vector
