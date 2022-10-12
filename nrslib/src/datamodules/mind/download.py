# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# https://github.com/Microsoft/Recommenders

import os
import random
import logging
import json
import numpy as np
import re
from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from ..components.utils import (
    maybe_download,
    download_path,
    unzip_file,
)


URL_MIND_LARGE_TRAIN = (
    "https://mind201910small.blob.core.windows.net/release/MINDlarge_train.zip"
)
URL_MIND_LARGE_VALID = (
    "https://mind201910small.blob.core.windows.net/release/MINDlarge_dev.zip"
)
URL_MIND_LARGE_TEST = (
    "https://mind201910small.blob.core.windows.net/release/MINDlarge_test.zip"
)
URL_MIND_LARGE_UTILS = (
    "https://mind201910small.blob.core.windows.net/release/MINDlarge_utils.zip"
)
URL_MIND_SMALL_TRAIN = (
    "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip"
)
URL_MIND_SMALL_VALID = (
    "https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip"
)
URL_MIND_SMALL_UTILS = (
    "https://mind201910small.blob.core.windows.net/release/MINDsmall_utils.zip"
)
URL_MIND_DEMO_TRAIN = (
    "https://recodatasets.z20.web.core.windows.net/newsrec/MINDdemo_train.zip"
)
URL_MIND_DEMO_VALID = (
    "https://recodatasets.z20.web.core.windows.net/newsrec/MINDdemo_dev.zip"
)
URL_MIND_DEMO_UTILS = (
    "https://recodatasets.z20.web.core.windows.net/newsrec/MINDdemo_utils.zip"
)


URL_MIND = {
    "large": (
        URL_MIND_LARGE_TRAIN,
        URL_MIND_LARGE_VALID,
        URL_MIND_LARGE_TEST,
        URL_MIND_LARGE_UTILS,
    ),
    "small": (
        URL_MIND_SMALL_TRAIN,
        URL_MIND_SMALL_VALID,
        URL_MIND_LARGE_TEST,
        URL_MIND_SMALL_UTILS,
    ),
    "demo": (
        URL_MIND_DEMO_TRAIN,
        URL_MIND_DEMO_VALID,
        URL_MIND_LARGE_TEST,
        URL_MIND_DEMO_UTILS,
    ),
}

URL_WIKIDATA_KG = "https://kredkg.blob.core.windows.net/wikidatakg/kg.zip"

logger = logging.getLogger()


def download_and_extract_mind(size="small", dest_path=None):
    """Download and extract the MIND dataset

        Args:
            size (str): Dataset size
            dest_path (str): Save path for the zip dataset

        Returns:
            Tuple (train_path, valid_path, test_path) where train_path is the path to the train folder, valid_path is the path to the validation folder and test_path is the path to the test folder
    """
    size_options = ["small", "large", "demo"]
    if size not in size_options:
        raise ValueError(f"Wrong size option, available options are {size_options}")
    url_train, url_valid, url_test, url_utils = URL_MIND[size]
    if dest_path is None:
        dest_path = os.getcwd()
    print(f"Downloading MIND to {dest_path}")
    with download_path(dest_path) as path:
        train_zip = maybe_download(url=url_train, work_directory=path)
        valid_zip = maybe_download(url=url_valid, work_directory=path)
        test_zip = (
            maybe_download(url=url_test, work_directory=path)
            if size == "large"
            else None
        )
        train_path, valid_path, test_path = extract_mind(
            train_zip, valid_zip, test_zip, root_folder=dest_path
        )
        # utils_path = maybe_download(url=url_utils, work_directory=path)
    return train_path, valid_path, test_path  # , utils_path


def download_and_extract_wikidata_kg(dest_path, clean_zip_file):
    """Download and extract the wikidata knowledge graph for the MIND dataset

        Args:
            dest_path (str): Path for saving the downloaded zip file
            clean_zip_file (bool): Whether to delete the zip file after unzipping

        Returns:
            Path to the unzipped wikidata knowledge graph folder
    """
    print(f"Downloading MIND wikidata knowledge graph")
    with download_path(dest_path) as path:
        kg_zip = maybe_download(url=URL_WIKIDATA_KG, work_directory=path)
        kg_path = os.path.join(dest_path, "train", "wikidata_kg")
        unzip_file(kg_zip, kg_path, clean_zip_file=clean_zip_file)
    return kg_path


def extract_mind(
    train_zip,
    valid_zip,
    test_zip,
    # utils_zip,
    root_folder=None,
    train_folder="train",
    valid_folder="valid",
    test_folder="test",
    # utils_folder="utils",
    clean_zip_file=False,
):
    """Extract MIND dataset

    Args:
        train_zip (str): Path to train zip file
        valid_zip (str): Path to valid zip file
        train_folder (str): Destination folder for train set
        valid_folder (str): Destination folder for validation set

    Returns:
        Tuple (path_train, path_valid) where path_train is the path to the training folder and path_valid is the path to the validation folder
    """
    root_folder = os.getcwd() if root_folder is None else root_folder
    train_path = os.path.join(root_folder, train_folder)
    valid_path = os.path.join(root_folder, valid_folder)
    test_path = None if test_zip is None else os.path.join(root_folder, test_folder)
    # utils_path = os.path.join(root_folder, utils_folder)
    unzip_file(train_zip, train_path, clean_zip_file=clean_zip_file)
    unzip_file(valid_zip, valid_path, clean_zip_file=clean_zip_file)
    if test_path is not None:
        unzip_file(test_zip, test_path, clean_zip_file=clean_zip_file)
    # unzip_file(utils_zip, utils_path, clean_zip_file=clean_zip_file)
    return train_path, valid_path, test_path  # , utils_path


def read_clickhistory(path, filename):
    """Read click history file

    Args:
        path (str): Folder path
        filename (str): Filename

    Returns:
        Tuple (list, dict) where list is a list of user session with user_id, clicks, positive and negative interactions and dict is a dictionary with user_id click history.
    """
    userid_history = {}
    with open(os.path.join(path, filename)) as f:
        lines = f.readlines()
    sessions = []
    for i in range(len(lines)):
        _, userid, imp_time, click, imps = lines[i].strip().split("\t")
        clicks = click.split(" ")
        pos = []
        neg = []
        imps = imps.split(" ")
        for imp in imps:
            if imp.split("-")[1] == "1":
                pos.append(imp.split("-")[0])
            else:
                neg.append(imp.split("-")[0])
        userid_history[userid] = clicks
        sessions.append([userid, clicks, pos, neg])
    return sessions, userid_history


def _newsample(nnn, ratio):
    if ratio > len(nnn):
        return random.sample(nnn * (ratio // len(nnn) + 1), ratio)
    else:
        return random.sample(nnn, ratio)


def get_train_input(session, train_file_path, npratio=4):
    """Generate train file.

    Args:
        session (list): List of user session with user_id, clicks, positive and negative interactions.
        train_file_path (str): Path to file.
        npratio (int): Ratio for negative sampling.
    """
    fp_train = open(train_file_path, "w", encoding="utf-8")
    for sess_id in range(len(session)):
        sess = session[sess_id]
        userid, _, poss, negs = sess
        for i in range(len(poss)):
            pos = poss[i]
            neg = _newsample(negs, npratio)
            fp_train.write("1 " + "train_" + userid + " " + pos + "\n")
            for neg_ins in neg:
                fp_train.write("0 " + "train_" + userid + " " + neg_ins + "\n")
    fp_train.close()
    if os.path.isfile(train_file_path):
        logger.info(f"Train file {train_file_path} successfully generated")
    else:
        raise FileNotFoundError(f"Error when generating {train_file_path}")


def get_valid_input(session, valid_file_path):
    """Generate validation file.

    Args:
        session (list): List of user session with user_id, clicks, positive and negative interactions.
        valid_file_path (str): Path to file.
    """
    fp_valid = open(valid_file_path, "w", encoding="utf-8")
    for sess_id in range(len(session)):
        userid, _, poss, negs = session[sess_id]
        for i in range(len(poss)):
            fp_valid.write(
                "1 " + "valid_" + userid + " " + poss[i] + "%" + str(sess_id) + "\n"
            )
        for i in range(len(negs)):
            fp_valid.write(
                "0 " + "valid_" + userid + " " + negs[i] + "%" + str(sess_id) + "\n"
            )
    fp_valid.close()
    if os.path.isfile(valid_file_path):
        logger.info(f"Validation file {valid_file_path} successfully generated")
    else:
        raise FileNotFoundError(f"Error when generating {valid_file_path}")


def get_user_history(train_history, valid_history, user_history_path):
    """Generate user history file.

    Args:
        train_history (list): Train history.
        valid_history (list): Validation history
        user_history_path (str): Path to file.
    """
    fp_user_history = open(user_history_path, "w", encoding="utf-8")
    for userid in train_history:
        fp_user_history.write(
            "train_" + userid + " " + ",".join(train_history[userid]) + "\n"
        )
    for userid in valid_history:
        fp_user_history.write(
            "valid_" + userid + " " + ",".join(valid_history[userid]) + "\n"
        )
    fp_user_history.close()
    if os.path.isfile(user_history_path):
        logger.info(f"User history file {user_history_path} successfully generated")
    else:
        raise FileNotFoundError(f"Error when generating {user_history_path}")

def read_news_ids(filepath):
    """ Read news ids

    Args:
        filepath (str): Path to news file

    Returns:
        Dictionary containing news identifiers and generated ids
    """
    news_ids = {}
    i = 0
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            splitted = line.strip("\n").split("\t")
            news_id = splitted[0]
            news_ids[news_id] = i
            i += 1
    return news_ids


def read_news(filepath, tokenizer):
    """Read news file

    Args:
        filepath (str): Path to news file
        tokenizer (tokenizer): Tokenizer for news title tokenization

    Returns:
        Tuple (news_words, news_entities, news_abstract_entities, news_categories, news_subcategories) where each item is a dictionary containing the items, specified by the dictionary name, assigned to each user
    """
    news_words = {}
    news_entities = {}
    news_abstract_entities = {}
    news_categories = {}
    news_subcategories = {}
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            splitted = line.strip("\n").split("\t")
            news_categories[splitted[0]] = splitted[1]
            news_subcategories[splitted[0]] = splitted[2]
            stop_words = set(stopwords.words("english"))
            word_tokens = tokenizer.tokenize(splitted[3].lower())
            news_words[splitted[0]] = [w for w in word_tokens if not w in stop_words]
            news_entities[splitted[0]] = []
            news_abstract_entities[splitted[0]] = []
            try:
                for entity in json.loads(splitted[6]):
                    news_entities[splitted[0]].append(
                        (entity["SurfaceForms"], entity["WikidataId"])
                    )
            except:
                pass
            try:
                for entity in json.loads(splitted[7]):
                    news_abstract_entities[splitted[0]].append(
                        (entity["SurfaceForms"], entity["WikidataId"])
                    )
            except:
                pass
    return (
        news_words,
        news_entities,
        news_abstract_entities,
        news_categories,
        news_subcategories,
    )


def download_and_extract_glove(zip_path=None, dest_path=None, glove_size=6):
    """Download and extract the Glove embedding

    Args:
        dest_path (str): Destination directory path for the downloaded file

    Returns:
        File path where Glove was extracted.
    """
    # url = f"http://nlp.stanford.edu/data/glove.{glove_size}B.zip"
    url = (
        f"https://huggingface.co/stanfordnlp/glove/resolve/main/glove.{glove_size}B.zip"
    )
    dest_path = os.getcwd() if dest_path is None else dest_path
    with download_path(zip_path) as path:
        filepath = maybe_download(url=url, work_directory=path)
        glove_path = os.path.join(dest_path, "glove")
        unzip_file(filepath, glove_path, clean_zip_file=False)
    return glove_path


def generate_embeddings(
    data_path,
    news_words,
    news_entities,
    train_entities,
    valid_entities,
    max_sentence=10,
    word_embedding_dim=100,
):
    """Generate embeddings.

    Args:
        data_path (str): Data path.
        news_words (dict): News word dictionary.
        news_entities (dict): News entity dictionary.
        train_entities (str): Train entity file.
        valid_entities (str): Validation entity file.
        max_sentence (int): Max sentence size.
        word_embedding_dim (int): Word embedding dimension.

    Returns:
        Tuple containing the paths to the news, word and entity embeddings
    """
    embedding_dimensions = [50, 100, 200, 300]
    if word_embedding_dim not in embedding_dimensions:
        raise ValueError(
            f"Wrong embedding dimension, available options are {embedding_dimensions}"
        )

    logger.info("Downloading glove...")
    glove_path = download_and_extract_glove(data_path)

    word_set = set()
    word_embedding_dict = {}
    entity_embedding_dict = {}

    logger.info(f"Loading glove with embedding dimension {word_embedding_dim}...")
    glove_file = "glove.6B." + str(word_embedding_dim) + "d.txt"
    fp_pretrain_vec = open(os.path.join(glove_path, glove_file), "r", encoding="utf-8")
    for line in fp_pretrain_vec:
        linesplit = line.split(" ")
        word_set.add(linesplit[0])
        word_embedding_dict[linesplit[0]] = np.asarray(list(map(float, linesplit[1:])))
    fp_pretrain_vec.close()

    logger.info("Reading train entities...")
    fp_entity_vec_train = open(train_entities, "r", encoding="utf-8")
    for line in fp_entity_vec_train:
        linesplit = line.split()
        entity_embedding_dict[linesplit[0]] = np.asarray(
            list(map(float, linesplit[1:]))
        )
    fp_entity_vec_train.close()

    logger.info("Reading valid entities...")
    fp_entity_vec_valid = open(valid_entities, "r", encoding="utf-8")
    for line in fp_entity_vec_valid:
        linesplit = line.split()
        entity_embedding_dict[linesplit[0]] = np.asarray(
            list(map(float, linesplit[1:]))
        )
    fp_entity_vec_valid.close()

    logger.info("Generating word and entity indexes...")
    word_dict = {}
    word_index = 1
    news_word_string_dict = {}
    news_entity_string_dict = {}
    entity2index = {}
    entity_index = 1
    for doc_id in news_words:
        news_word_string_dict[doc_id] = [0 for n in range(max_sentence)]
        news_entity_string_dict[doc_id] = [0 for n in range(max_sentence)]
        surfaceform_entityids = news_entities[doc_id]
        for item in surfaceform_entityids:
            if item[1] not in entity2index and item[1] in entity_embedding_dict:
                entity2index[item[1]] = entity_index
                entity_index = entity_index + 1
        for i in range(len(news_words[doc_id])):
            if news_words[doc_id][i] in word_embedding_dict:
                if news_words[doc_id][i] not in word_dict:
                    word_dict[news_words[doc_id][i]] = word_index
                    word_index = word_index + 1
                    news_word_string_dict[doc_id][i] = word_dict[news_words[doc_id][i]]
                else:
                    news_word_string_dict[doc_id][i] = word_dict[news_words[doc_id][i]]
                for item in surfaceform_entityids:
                    for surface in item[0]:
                        for surface_word in surface.split(" "):
                            if news_words[doc_id][i] == surface_word.lower():
                                if item[1] in entity_embedding_dict:
                                    news_entity_string_dict[doc_id][i] = entity2index[
                                        item[1]
                                    ]
            if i == max_sentence - 1:
                break

    logger.info("Generating word embeddings...")
    word_embeddings = np.zeros([word_index, word_embedding_dim])
    for word in word_dict:
        word_embeddings[word_dict[word]] = word_embedding_dict[word]

    logger.info("Generating entity embeddings...")
    entity_embeddings = np.zeros([entity_index, word_embedding_dim])
    for entity in entity2index:
        entity_embeddings[entity2index[entity]] = entity_embedding_dict[entity]

    news_feature_path = os.path.join(data_path, "doc_feature.txt")
    logger.info(f"Saving word and entity features in {news_feature_path}")
    fp_doc_string = open(news_feature_path, "w", encoding="utf-8")
    for doc_id in news_word_string_dict:
        fp_doc_string.write(
            doc_id
            + " "
            + ",".join(list(map(str, news_word_string_dict[doc_id])))
            + " "
            + ",".join(list(map(str, news_entity_string_dict[doc_id])))
            + "\n"
        )

    word_embeddings_path = os.path.join(
        data_path, "word_embeddings_5w_" + str(word_embedding_dim) + ".npy"
    )
    logger.info(f"Saving word embeddings in {word_embeddings_path}")
    np.save(word_embeddings_path, word_embeddings)

    entity_embeddings_path = os.path.join(
        data_path, "entity_embeddings_5w_" + str(word_embedding_dim) + ".npy"
    )
    logger.info(f"Saving word embeddings in {entity_embeddings_path}")
    np.save(entity_embeddings_path, entity_embeddings)

    return news_feature_path, word_embeddings_path, entity_embeddings_path


def load_glove_matrix(path_emb, word_dict, word_embedding_dim):
    """Load pretrained embedding metrics of words in word_dict

    Args:
        path_emb (string): Folder path of downloaded glove file
        word_dict (dict): Word dictionary
        word_embedding_dim: Dimention of word embedding vectors

    Returns:
        Numpy.ndarray list containing pretrained word embedding metrics, words can be found in glove files
    """

    embedding_matrix = np.zeros((len(word_dict) + 1, word_embedding_dim))
    exist_word = []

    with open(os.path.join(path_emb, f"glove.6B.{word_embedding_dim}d.txt"), "rb") as f:
        for l in tqdm(f):  # noqa: E741 ambiguous variable name 'l'
            l = l.split()  # noqa: E741 ambiguous variable name 'l'
            word = l[0].decode()
            if len(word) != 0:
                if word in word_dict:
                    wordvec = [float(x) for x in l[1:]]
                    index = word_dict[word]
                    embedding_matrix[index] = np.array(wordvec)
                    exist_word.append(word)

    return embedding_matrix, exist_word


def word_tokenize(sent):
    """Tokenize a sentence

    Args:
        sent (str): Sentence to be tokenized

    Returns:
        Word list
    """

    # treat consecutive words or special punctuation as words
    pat = re.compile(r"[\w]+|[.,!?;|]")
    if isinstance(sent, str):
        return pat.findall(sent.lower())
    else:
        return []


