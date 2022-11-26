import numpy as np
import torch
from gensim.models import word2vec
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType, ModelType


class Item2Vec(GeneralRecommender):
    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        super(Item2Vec, self).__init__(config, dataset)

        # load parameters info
        self.vector_size = getattr(config, "vector_size", 128)
        self.window = getattr(config, "window", 5)
        self.epochs = getattr(config, "item2vec_epochs", 10)
        self.ns_exponent = getattr(config, "ns_exponent", 0.75)
        self.min_count = getattr(config, "min_count", 5)
        self.seed = getattr(config, "seed", 0)
        self.workers = getattr(config, "worker", 1)

        self.normalize = getattr(config, "normalize", True)
        self.dataset_item_num = dataset.item_num

        self.fake_loss = torch.nn.Parameter(torch.zeros(1))
        self.other_parameter_name = [
            "user_history",
            "item2vec",
            "all_item_vectors",
            "vector_size",
            "epochs",
            "ns_exponent",
            "seed",
            "min_count",
            "workers",
            "normalize",
        ]

        pretrained = getattr(config, "pretrained", False)
        if not pretrained:
            self.user_history = {}
            user_history_matrix, _, user_history_length = dataset.history_item_matrix()
            iters = enumerate(zip(user_history_matrix, user_history_length))
            for user_id, (item_ids, n_items) in iters:
                if n_items > 0:
                    self.user_history[user_id] = item_ids[:n_items].tolist()

            self.item2vec = word2vec.Word2Vec(
                self.user_history.values(),
                vector_size=self.vector_size,
                window=self.window,
                epochs=self.epochs,
                ns_exponent=self.ns_exponent,
                seed=self.seed,
                min_count=self.min_count,
                workers=self.workers,
            )

            self.all_item_vectors = self._get_all_item_embedding()

    def forward(self, user, item):
        pass

    def calculate_loss(self, interaction):
        return torch.nn.Parameter(torch.zeros(1))

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        return torch.mul(user_e, item_e).sum(dim=1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.all_item_vectors
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)

    def get_user_embedding(self, user_ids):
        default_vector = self._normalize(self.item2vec.wv.vectors).mean(axis=0)
        user_vectors = np.zeros((len(user_ids), self.vector_size))
        for i, user_id in enumerate(user_ids):
            if user_id in self.user_history:
                user_vectors[i, :] = self.item2vec.wv.get_mean_vector(
                    self.user_history[user_id],
                    pre_normalize=self.normalize,
                )
            else:
                user_vectors[i, :] = default_vector
        return torch.FloatTensor(user_vectors)

    def get_item_embedding(self, item_ids):
        item_vectors = self.item2vec.wv.vectors[item_ids]
        if self.normalize:
            item_vectors = self._normalize(item_vectors)
        return torch.FloatTensor(item_vectors)

    def _get_all_item_embedding(self):
        default_vector = self._normalize(self.item2vec.wv.vectors).mean(axis=0)
        item2vec_vocab = set(self.item2vec.wv.index_to_key)
        item_vectors = np.zeros((self.dataset_item_num, self.vector_size))
        for i in range(self.dataset_item_num):
            if i in item2vec_vocab:
                item_vectors[i, :] = self.item2vec.wv.get_vector(i)
            else:
                item_vectors[i, :] = default_vector
        return torch.FloatTensor(item_vectors)

    def _normalize(self, vectors):
        norm = np.linalg.norm(vectors, axis=1, keepdims=True)
        norm[norm == 0] = 1
        return vectors / norm
