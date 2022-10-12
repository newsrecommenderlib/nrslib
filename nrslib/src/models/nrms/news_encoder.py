import torch
import torch.nn as nn
import torch.nn.functional as F
from ..components.attention.multihead_self import MultiHeadSelfAttention
from ..components.attention.additive import AdditiveAttention


class NewsEncoder(torch.nn.Module):
    def __init__(
        self,
        num_words,
        word_embedding_dim,
        num_attention_heads,
        query_vector_dim,
        dropout_probability,
        pretrained_word_embedding,
    ):
        super(NewsEncoder, self).__init__()
        if pretrained_word_embedding is None:
            self.word_embedding = nn.Embedding(
                num_words, word_embedding_dim, padding_idx=0
            )
        else:
            self.word_embedding = nn.Embedding.from_pretrained(
                pretrained_word_embedding, freeze=False, padding_idx=0
            )

        self.multihead_self_attention = MultiHeadSelfAttention(
            word_embedding_dim, num_attention_heads
        )
        self.additive_attention = AdditiveAttention(
            query_vector_dim, word_embedding_dim
        )
        self.dropout_probability = dropout_probability

    def forward(self, news):
        """
        Args:
            news:
                {
                    "title": batch_size * num_words_title
                }
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        # batch_size, num_words_title, word_embedding_dim
        news_vector = F.dropout(
            self.word_embedding(news["title"]),
            p=self.dropout_probability,
            training=self.training,
        )
        # batch_size, num_words_title, word_embedding_dim
        multihead_news_vector = self.multihead_self_attention(news_vector)
        multihead_news_vector = F.dropout(
            multihead_news_vector, p=self.dropout_probability, training=self.training
        )
        # batch_size, word_embedding_dim
        final_news_vector = self.additive_attention(multihead_news_vector)
        return final_news_vector
