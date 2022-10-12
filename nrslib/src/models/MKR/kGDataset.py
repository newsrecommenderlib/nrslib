import os
import numpy as np

class KGDataset:
    """Knowledge graph dataset for MKR model

    Args:
        path (str): Path to knowledge graph file
    """
    def __init__(self, path):
        self.n_entity, self.n_relation, self.kg = self._load_kg(path)

    def __getitem__(self, index):
        return self.kg[index]

    def __len__(self):
        return len(self.kg)

    def _load_kg(self, path):
        """Load knowledge graph

        Args:
            path (str): Path to knowledge graph file

        Returns:
            A tuple (n_entity, n_relation, kg) where n_entity is the number of entities, n_relation is the number of relations and kg is the knowledge graph

        """
        print('Loading knowledge graph file for model')
        kg_file = os.path.join(path, r'MKR', r'kg_final')
        if os.path.exists(kg_file + '.npy'):
            kg = np.load(kg_file + '.npy')
        else:
            kg = np.loadtxt(kg_file + '.txt', dtype=np.int32)
            np.save(kg_file + '.npy', kg)
        n_entity = len(set(kg[:, 0]) | set(kg[:, 2]))
        n_relation = len(set(kg[:, 1]))

        return n_entity, n_relation, kg