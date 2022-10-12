import colorlog
from os.path import join
import os
from tqdm.auto import tqdm
from .download import (
    read_news,
    read_clickhistory,
)
import numpy as np
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict


def create_knowledge_graph_file(
        model,
        paths,
        use_categories,
        use_subcategories,
        use_title_entities,
        use_abstract_entities,
        use_title_tokens,
        use_wikidata,
):
    """Creates a news article knowledge graph file including the properties set in the constructor.

    Args:
        model (str): Model name, either "MKR" or "RippleNet". Necessary for filepath.
        paths (list): Contains paths to the train and/or validation and/or test news files.
        use_categories (boolean): Whether to use news categories in knowledge graph.
        use_subcategories (boolean): Whether to use news subcategories in knowledge graph.
        use_title_entities (boolean): Whether to use news title entities in knowledge graph.
        use_abstract_entities (boolean): Whether to use news abstract entities in knowledge graph.
        use_title_tokens (boolean): Whether to use news title tokens in knowledge graph.
        use_wikidata (boolean): Whether to use additional wikidata knowledge graph in knowledge graph.

    Returns:
        Tuple (path1, path2) where path1 is the path to the file containing the knowledge graph and path2 is the path to the file containing the item index to entity id hashes.

    """
    folder_path = ""
    for path in paths:
        if path is None:
            paths.remove(path)
    for path in paths:
        if "train" in path:
            folder_path = os.path.join(path, model)
    if os.path.exists(os.path.join(folder_path, r"kg_final.txt")):
        colorlog.warning(
            f"the model now uses a pre-existing knowledge graph file. Make sure it contains the appropriate data."
            f"If not delete the following files before running the model again: \n"
            f"{os.path.join(path, model, r'kg_final.txt')} and {os.path.join(path, model, r'item_index2entity_id_rehashed.txt')}")
        return (os.path.join(folder_path, r"kg_final.txt")), (
            os.path.join(folder_path, r"item_index2entity_id_rehashed.txt"))
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    tokenizer = RegexpTokenizer(r"\w+")
    list_categories = {}
    list_subcategories = {}
    list_title_tokens = {}
    list_news = []
    list_entities = {}
    list_relationships = []

    i = 0
    for path in paths:
        print("knowledge graph preparation..." + str(path))
        (
            dict_title_tokens,
            dict_news_title_entities,
            dict_news_abstract_entities,
            dict_news_categories,
            dict_news_subcategories,
        ) = read_news(join(path, r"news.tsv"), tokenizer)
        if use_categories:
            for key in dict_news_categories:
                if key not in list_news:
                    list_news.append(key)
                if dict_news_categories[key] not in list_categories:
                    list_categories[dict_news_categories[key]] = i
                    i += 1
        if use_subcategories:
            for key in dict_news_subcategories:
                if dict_news_subcategories[key] not in list_subcategories:
                    list_subcategories[dict_news_subcategories[key]] = i
                    i += 1
        if use_title_tokens:
            for key in dict_title_tokens:
                for token in dict_title_tokens[key]:
                    if token not in list_title_tokens:
                        list_title_tokens[token] = i
                        i += 1
        if use_title_entities:
            for key in dict_news_title_entities:
                for entity in dict_news_title_entities[key]:
                    if entity[1] not in list_entities:
                        list_entities[entity[1]] = i
                        i += 1
        if use_abstract_entities:
            for key in dict_news_abstract_entities:
                for entity in dict_news_abstract_entities[key]:
                    if entity[1] not in list_entities:
                        list_entities[entity[1]] = i
                        i += 1

    list_written_news = []
    num_news = len(list_news)

    for path in paths:
        print("creating knowledge graph file...")

        (
            dict_title_tokens,
            dict_news_title_entities,
            dict_news_abstract_entities,
            dict_news_categories,
            dict_news_subcategories,
        ) = read_news(join(path, r"news.tsv"), tokenizer)

        with open(os.path.join(folder_path, r"kg_final.txt"), "a") as f1:
            for key in tqdm(dict_title_tokens):
                if key in list_written_news:
                    pass
                else:
                    list_written_news.append(key)
                    news_id = list_news.index(key)

                    if use_categories:
                        category = dict_news_categories[key]
                        category_index = num_news + list_categories[category]
                        relationship = "news.article.category"
                        if relationship not in list_relationships:
                            list_relationships.append(relationship)
                        relationship_index = list_relationships.index(relationship)
                        line = (
                                str(news_id)
                                + "\t"
                                + str(relationship_index)
                                + "\t"
                                + str(category_index)
                        )
                        f1.write(line)
                        f1.write("\n")

                        if model == "rippleNet":
                            relationship = "news.category.articles"
                            if relationship not in list_relationships:
                                list_relationships.append(relationship)
                            relationship_index = list_relationships.index(relationship)
                            line = (
                                    str(category_index)
                                    + "\t"
                                    + str(relationship_index)
                                    + "\t"
                                    + str(news_id)
                            )
                            f1.write(line)
                            f1.write("\n")

                    if use_subcategories:
                        subcategory = dict_news_subcategories[key]
                        subcategory_index = num_news + list_subcategories[subcategory]
                        relationship = "news.article.subcategory"
                        if relationship not in list_relationships:
                            list_relationships.append(relationship)
                        relationship_index = list_relationships.index(relationship)
                        line = (
                                str(news_id)
                                + "\t"
                                + str(relationship_index)
                                + "\t"
                                + str(subcategory_index)
                        )
                        f1.write(line)
                        f1.write("\n")

                        if model == "rippleNet":
                            relationship = "news.subcategory.articles"
                            if relationship not in list_relationships:
                                list_relationships.append(relationship)
                            relationship_index = list_relationships.index(relationship)
                            line = (
                                    str(subcategory_index)
                                    + "\t"
                                    + str(relationship_index)
                                    + "\t"
                                    + str(news_id)
                            )
                            f1.write(line)
                            f1.write("\n")

                    if use_title_tokens:
                        for token in dict_title_tokens[key]:
                            title_token_index = num_news + list_title_tokens[token]
                            relationship = "news.article.token"
                            if relationship not in list_relationships:
                                list_relationships.append(relationship)
                            relationship_index = list_relationships.index(relationship)
                            line = (
                                    str(news_id)
                                    + "\t"
                                    + str(relationship_index)
                                    + "\t"
                                    + str(title_token_index)
                            )
                            f1.write(line)
                            f1.write("\n")

                            if model == "rippleNet":
                                relationship = "news.token.articles"
                                if relationship not in list_relationships:
                                    list_relationships.append(relationship)
                                relationship_index = list_relationships.index(
                                    relationship
                                )
                                line = (
                                        str(title_token_index)
                                        + "\t"
                                        + str(relationship_index)
                                        + "\t"
                                        + str(news_id)
                                )
                                f1.write(line)
                                f1.write("\n")

                    if use_title_entities:
                        for entity in dict_news_title_entities[key]:
                            entity_index = num_news + list_entities[entity[1]]
                            relationship = "news.article.entity"
                            if relationship not in list_relationships:
                                list_relationships.append(relationship)
                            relationship_index = list_relationships.index(relationship)
                            line = (
                                    str(news_id)
                                    + "\t"
                                    + str(relationship_index)
                                    + "\t"
                                    + str(entity_index)
                            )
                            f1.write(line)
                            f1.write("\n")

                            if model == "rippleNet":
                                relationship = "news.entity.articles"
                                if relationship not in list_relationships:
                                    list_relationships.append(relationship)
                                relationship_index = list_relationships.index(
                                    relationship
                                )
                                line = (
                                        str(entity_index)
                                        + "\t"
                                        + str(relationship_index)
                                        + "\t"
                                        + str(news_id)
                                )
                                f1.write(line)
                                f1.write("\n")

                    if use_abstract_entities:
                        for entity in dict_news_abstract_entities[key]:
                            entity_index = num_news + list_entities[entity[1]]
                            relationship = "news.article.abstract_entity"
                            if relationship not in list_relationships:
                                list_relationships.append(relationship)
                            relationship_index = list_relationships.index(relationship)
                            line = (
                                    str(news_id)
                                    + "\t"
                                    + str(relationship_index)
                                    + "\t"
                                    + str(entity_index)
                            )
                            f1.write(line)
                            f1.write("\n")

                            if model == "rippleNet":
                                relationship = "news.abstract_entity.articles"
                                if relationship not in list_relationships:
                                    list_relationships.append(relationship)
                                relationship_index = list_relationships.index(
                                    relationship
                                )
                                line = (
                                        str(entity_index)
                                        + "\t"
                                        + str(relationship_index)
                                        + "\t"
                                        + str(news_id)
                                )
                                f1.write(line)
                                f1.write("\n")

    if use_wikidata:
        list_entities_new = {}
        i = max(list_entities.values()) + 1
        list_relationships_new = []
        print("wikidata knowledge graph creation...")
        lines = []
        with open(os.path.join(folder_path, "wikidata_kg", "wikidata-graph.tsv"), encoding="utf-8") as f:
            for line in f:
                splitted = line.strip("\n").split("\t")
                wikidata_id_start = splitted[0]
                wikidata_id_end = splitted[2]
                wikidata_id_relationship = splitted[1]
                start_id = ""
                relationship_id = ""
                end_id = ""

                if wikidata_id_start in list_entities or wikidata_id_end in list_entities:
                    try:
                        start_id = list_entities[wikidata_id_start]
                    except:
                        list_entities_new[wikidata_id_start] = i
                        start_id = list_entities_new[wikidata_id_start]
                        i += 1
                    try:
                        end_id = list_entities[wikidata_id_end]
                    except:
                        list_entities_new[wikidata_id_end] = i
                        end_id = list_entities_new[wikidata_id_end]
                        i += 1

                    if wikidata_id_relationship not in list_relationships_new:
                        list_relationships_new.append(wikidata_id_relationship)
                    relationship_id = list_relationships.index(wikidata_id_relationship)

                    line = (str(start_id) + "\t" + str(relationship_id) + "\t" + str(end_id))
                    lines.append(line)

                    if model == "rippleNet":
                        relationship = "-" + str(wikidata_id_relationship)
                        if relationship not in list_relationships:
                            list_relationships.append(relationship)
                        relationship_id = list_relationships.index(relationship)
                        line = (str(end_id) + "\t" + str(relationship_id) + "\t" + str(start_id))
                        lines.append(line)

        with open(os.path.join(folder_path, r"kg_final.txt"), "a") as f1:
            for line in tqdm(lines):
                f.write(line)
                f.write("\n")
                i += 1

    with open(
            os.path.join(folder_path, r"item_index2entity_id_rehashed.txt"), "w"
    ) as f:
        i = 0
        for item in list_news:
            f.write(str(item) + "\t" + str(i))
            f.write("\n")
            i += 1

    return (os.path.join(folder_path, r"kg_final.txt")), (
        os.path.join(folder_path, r"item_index2entity_id_rehashed.txt"))


def create_rating_file(paths, path_to_news_ids, model):
    """Creates a file specifying which news articles have been read by users and which have not

    Args:
        paths (list): List containing paths to behaviours files (train, validation, test)
        path_to_news_ids (str): Path to file containing news ids and corresponding item index
        model (str): Model name, either "MKR" or "RippleNet". Necessary for path

    """

    for path in paths:
        if path is None:
            paths.remove(path)

    for path in paths:
        folder_path = os.path.join(path, model)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    user_ids = {}
    user_ids_to_int = {}
    i = 0
    for path in paths:
        folder_path = os.path.join(path, model)
        if os.path.exists(os.path.join(folder_path, r"ratings_final.txt")):
            warning_text = ""
            for path in paths:
                warning_text = warning_text + f" {os.path.join(path, model, r'ratings_final.txt')} and {os.path.join(path, model, r'user2int.txt')} \n"
            colorlog.warning(
                f"the model now uses pre-existing ratings files. Make sure they contain the appropriate data. "
                f"If not delete the following files before running the model again: \n" + warning_text)
            return
        print("creating rating file...")
        sessions, userid_history = read_clickhistory(path, r"behaviors.tsv")

        with open(path_to_news_ids, encoding="utf-8", ) as f:
            lines = f.readlines()
        news_ids = {}
        for line in lines:
            splitted = line.strip("\n").split("\t")
            news_ids[splitted[0]] = splitted[1]

        positives = defaultdict(set)
        negatives = defaultdict(set)
        session_id = 0
        if "train" in folder_path:
            for element in sessions:
                user_id = element[0]
                if user_id not in user_ids_to_int:
                    user_ids_to_int[user_id] = i
                    i += 1
                user_ids[session_id] = user_ids_to_int[user_id]
                item_set = set()
                for item in element[1]:
                    item_set.add(item)
                for item in element[2]:
                    item_set.add(item)
                positives[session_id] |= item_set
                item_set = set()
                for item in element[3]:
                    item_set.add(item)
                negatives[session_id] |= item_set
                session_id += 1
        else:
            for element in sessions:
                user_id = element[0]
                if user_id not in user_ids_to_int:
                    user_ids_to_int[user_id] = i
                    i += 1
                user_ids[session_id] = user_ids_to_int[user_id]
                item_set = set()
                for item in element[2]:
                    item_set.add(item)
                positives[session_id] |= item_set
                item_set = set()
                for item in element[3]:
                    item_set.add(item)
                negatives[session_id] |= item_set
                session_id += 1

        with open(os.path.join(folder_path, r"ratings_final.txt"), 'w') as f:
            for key in user_ids:
                for element in positives[key]:
                    if element != "":
                        if "train" in folder_path:
                            line = str(user_ids[key]) + "\t" + str(news_ids[element]) + "\t" + str(1)
                        else:
                            line = str(user_ids[key]) + "\t" + str(news_ids[element]) + "\t" + str(1) + "\t" + str(key)
                        f.write(
                            line
                        )
                        f.write("\n")
                for element in negatives[key]:
                    if element != "":
                        line = ""
                        if "train" in folder_path:
                            line = str(user_ids[key]) + "\t" + str(news_ids[element]) + "\t" + str(0)
                        else:
                            line = str(user_ids[key]) + "\t" + str(news_ids[element]) + "\t" + str(0) + "\t" + str(key)
                        f.write(
                            line
                        )
                        f.write("\n")
    for path in paths:
        with open(os.path.join(path, model, r"user2int.txt"), "w") as f:
            for key in user_ids_to_int:
                f.write(str(key) + "\t" + str(user_ids_to_int[key]))
                f.write("\n")


def prepare_numpy_data(path, model):
    """Converts the rating.txt file to numpy format

    Args:
        path (str): Path to the ratings.txt file
        model (str): Model name, either "MKR" or "RippleNet". Necessary for path

    Returns:
        Numpy.ndarray containing the rating data in numpy format

    """
    rating_file = os.path.join(path, model, r"ratings_final")
    rating_np = np.loadtxt(rating_file + ".txt", dtype=np.int32)
    np.save(rating_file + ".npy", rating_np)
    return rating_np


def prepare_numpy_kg(path, model):
    """Converts the knowledge graph.txt file to numpy format

    Args:
        path (str): Path to the knowledge graph.txt file
        model (str): Model name, either "MKR" or "RippleNet". Necessary for path

    Returns:
        numpy.Ndarray containing the knowledge graph data in numpy format

    """
    kg_file = os.path.join(path, model, r"kg_final")
    kg_np = np.loadtxt(kg_file + ".txt", dtype=np.int32)
    np.save(kg_file + ".npy", kg_np)
    return kg_np


def create_rating_file_collaborative(paths, model):
    """Creates a file specifying which news articles have been read by users and which have not. Exclusively for the Collaborative Filtering model

    Args:
        paths (list): List containing paths to behaviours files (train, validation, test)
        model (str): Model name, either "MKR" or "RippleNet". Necessary for path

    """

    for path in paths:
        if path is None:
            paths.remove(path)

    for path in paths:
        folder_path = os.path.join(path, model)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    user_ids_to_int = {}
    user_ids = {}
    i = 0
    news_ids = {}
    n = 0
    for path in paths:

        folder_path = os.path.join(path, model)
        if os.path.exists(os.path.join(folder_path, r"ratings_final.txt")):
            warning_text = ""
            for path1 in paths:
                warning_text = warning_text + f"{os.path.join(path1, model, r'ratings_final.txt')} and {os.path.join(path, model, r'user2int.txt')} and {os.path.join(path, model, r'item_index2entity_id_rehashed.txt')} \n"
            colorlog.warning(
                f"the model now uses pre-existing ratings files. Make sure they contain the appropriate data. "
                f"If not delete the following files before running the model again: \n" + warning_text)
            return
        print("creating rating file...")
        sessions, userid_history = read_clickhistory(path, r"behaviors.tsv")

        positives = defaultdict(set)
        negatives = defaultdict(set)
        session_id = 0
        if "train" in folder_path:
            for element in sessions:
                user_id = element[0]
                if user_id not in user_ids_to_int:
                    user_ids_to_int[user_id] = i
                    i += 1
                user_ids[session_id] = user_ids_to_int[user_id]
                item_set = set()
                for item in element[1]:
                    item_set.add(item)
                    if item not in news_ids:
                        news_ids[item] = n
                        n += 1
                for item in element[2]:
                    item_set.add(item)
                    if item not in news_ids:
                        news_ids[item] = n
                        n += 1
                positives[session_id] |= item_set
                item_set = set()
                for item in element[3]:
                    item_set.add(item)
                    if item not in news_ids:
                        news_ids[item] = n
                        n += 1
                negatives[session_id] |= item_set
                session_id += 1
        else:
            for element in sessions:
                user_id = element[0]
                if user_id not in user_ids_to_int:
                    user_ids_to_int[user_id] = i
                    i += 1
                user_ids[session_id] = user_ids_to_int[user_id]
                item_set = set()
                for item in element[2]:
                    item_set.add(item)
                    if item not in news_ids:
                        news_ids[item] = n
                        n += 1
                positives[session_id] |= item_set
                item_set = set()
                for item in element[3]:
                    item_set.add(item)
                    if item not in news_ids:
                        news_ids[item] = n
                        n += 1
                negatives[session_id] |= item_set
                session_id += 1

        with open(os.path.join(folder_path, r"ratings_final.txt"), 'w') as f:
            for key in user_ids:
                for element in positives[key]:
                    if element != "":
                        if "train" in folder_path:
                            line = str(user_ids[key]) + "\t" + str(news_ids[element]) + "\t" + str(1)
                        else:
                            line = str(user_ids[key]) + "\t" + str(news_ids[element]) + "\t" + str(1) + "\t" + str(key)
                        f.write(
                            line
                        )
                        f.write("\n")
                for element in negatives[key]:
                    if element != "":
                        line = ""
                        if "train" in folder_path:
                            line = str(user_ids[key]) + "\t" + str(news_ids[element]) + "\t" + str(0)
                        else:
                            line = str(user_ids[key]) + "\t" + str(news_ids[element]) + "\t" + str(0) + "\t" + str(key)
                        f.write(
                            line
                        )
                        f.write("\n")

    for path in paths:
        with open(os.path.join(path, model, r"user2int.txt"), "w") as f:
            for key in user_ids_to_int:
                f.write(str(key) + "\t" + str(user_ids_to_int[key]))
                f.write("\n")

        with open(os.path.join(path, model, r"item_index2entity_id_rehashed.txt"), "w") as f:
            for key in news_ids:
                f.write(str(key) + "\t" + str(news_ids[key]))
                f.write("\n")
