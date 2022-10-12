import torch
import torch.nn as nn
import torch.nn.functional as F
from ..components.attention.additive import AdditiveAttention



class KCNN(torch.nn.Module):
    """
    Knowledge-aware CNN (KCNN) based on Kim CNN.
    Input a news sentence (e.g. its title), produce its embedding vector.
    """
    def __init__(self, pretrained_word_embedding,
                 pretrained_entity_embedding,
                 pretrained_context_embedding,
                 num_words,
                 word_embedding_dim,
                 use_context,
                 num_entities,
                 entity_embedding_dim,num_filters,window_sizes,query_vector_dim):
        super(KCNN, self).__init__()
        self.num_words=num_words
        self.word_embedding_dim=word_embedding_dim
        self.num_entities=num_entities,
        self.entity_embedding_dim=entity_embedding_dim
        self.use_context=use_context
        self.num_filters=num_filters
        self.window_sizes=window_sizes
        self.query_vector_dim=query_vector_dim
        if pretrained_word_embedding is None:
            self.word_embedding = nn.Embedding(self.num_words,
                                               self.word_embedding_dim,
                                               padding_idx=0)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(
                pretrained_word_embedding, freeze=False, padding_idx=0)
        if pretrained_entity_embedding is None:
            self.entity_embedding = nn.Embedding(num_entities,
                                                 self.entity_embedding_dim,
                                                 padding_idx=0)
        else:
            self.entity_embedding = nn.Embedding.from_pretrained(
                pretrained_entity_embedding, freeze=False, padding_idx=0)
        if self.use_context:
            if pretrained_context_embedding is None:
                self.context_embedding = nn.Embedding(
                    self.num_entities,
                    self.entity_embedding_dim,
                    padding_idx=0)
            else:
                self.context_embedding = nn.Embedding.from_pretrained(
                    pretrained_context_embedding, freeze=False, padding_idx=0)
        self.transform_matrix = nn.Parameter(
            torch.empty(self.entity_embedding_dim,
                        self.word_embedding_dim).uniform_(-0.1, 0.1))
        self.transform_bias = nn.Parameter(
            torch.empty(self.word_embedding_dim).uniform_(-0.1, 0.1))

        self.conv_filters = nn.ModuleDict({
            str(x): nn.Conv2d(3 if self.use_context else 2,
                              self.num_filters,
                              (x, self.word_embedding_dim))
            for x in self.window_sizes
        })
        self.additive_attention = AdditiveAttention(
            self.query_vector_dim, self.num_filters)

    def forward(self, news):
        """
        Args:
          news:
            {
                "title": batch_size * num_words_title,
                "title_entities": batch_size * num_words_title
            }

        Returns:
            final_vector: batch_size, len(window_sizes) * num_filters
        """
        # batch_size, num_words_title, word_embedding_dim
        word_vector = self.word_embedding(news["title"])
        # batch_size, num_words_title, entity_embedding_dim
        entity_vector = self.entity_embedding(
            news["title_entities"])
        if self.use_context:
            # batch_size, num_words_title, entity_embedding_dim
            context_vector = self.context_embedding(
                news["title_entities"])

        # batch_size, num_words_title, word_embedding_dim
        transformed_entity_vector = torch.tanh(
            torch.add(torch.matmul(entity_vector, self.transform_matrix),
                      self.transform_bias))

        if self.use_context:
            # batch_size, num_words_title, word_embedding_dim
            transformed_context_vector = torch.tanh(
                torch.add(torch.matmul(context_vector, self.transform_matrix),
                          self.transform_bias))

            # batch_size, 3, num_words_title, word_embedding_dim
            multi_channel_vector = torch.stack([
                word_vector, transformed_entity_vector,
                transformed_context_vector
            ],
                                               dim=1)
        else:
            # batch_size, 2, num_words_title, word_embedding_dim
            multi_channel_vector = torch.stack(
                [word_vector, transformed_entity_vector], dim=1)

        pooled_vectors = []
        for x in self.window_sizes:
            # batch_size, num_filters, num_words_title + 1 - x
            convoluted = self.conv_filters[str(x)](
                multi_channel_vector).squeeze(dim=3)
            # batch_size, num_filters, num_words_title + 1 - x
            activated = F.relu(convoluted)
            # batch_size, num_filters
            # Here we use a additive attention module
            # instead of pooling in the paper
            pooled = self.additive_attention(activated.transpose(1, 2))
            # pooled = activated.max(dim=-1)[0]
            # # or
            # # pooled = F.max_pool1d(activated, activated.size(2)).squeeze(dim=2)
            pooled_vectors.append(pooled)
        # batch_size, len(window_sizes) * num_filters
        final_vector = torch.cat(pooled_vectors, dim=1)
        return final_vector
