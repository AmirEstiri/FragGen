import json
import os
import numpy as np


def find_index_from_id(id, data):
    for i, d in enumerate(data):
        if id == d[0]:
            return i
    return -1


def partial_match_score(str1, str2):
    score = 0
    for w1 in str1.split(' '):
        for w2 in str2.split(' '):
            if w1.lower() == w2.lower():
                score += 1
    return score


def find_frag_id(frag_name, designer_name):
    """
    Find the id of a fragrance using its name and designer
    """
    f = open("data/dataset/initial_dataset.json")
    dataset = json.load(f)
    f.close()

    max_score = 0
    best_match = -1
    for data in dataset:
        if data[2] == designer_name:
            score = partial_match_score(frag_name, data[1])
            if score > max_score:
                max_score = score
                best_match = data[0]
    return best_match

def create_initial_graph_dataset():
    """
    Prepare data in a graph structure
    """
    
    ACCORDS = []
    NOTES = []
    dataset = []
    
    for id, f_name in enumerate(os.listdir("data/fragrances/")):
        f = open(f"data/fragrances/{f_name}")
        frag_data = json.load(f)
        f.close()

        name = frag_data['name']
        designer = frag_data['designer']
        
        if frag_data['sex']=="for men":
            sex = 0
        if frag_data['sex']=="for women":
            sex = 1
        else:
            sex = 2
        
        accords = list(frag_data['accords'].keys())
        accords_values = list(frag_data['accords'].values())
        for accord in accords:
            if accord not in ACCORDS:
                ACCORDS.append(accord)

        
        notes = frag_data['notes']['all'] + frag_data['notes']['top'] + frag_data['notes']['middle'] + frag_data['notes']['base']
        for note in notes:
            if note not in NOTES:
                NOTES.append(note)
        
        rating = frag_data['rating']
        total_votes = frag_data['total_votes']
        similar = frag_data['similar_fragrances']

        dataset.append([id, name, designer, sex, accords, accords_values, notes, rating, total_votes, similar])

    f = open("data/dataset/initial_dataset.json", "w")
    f.write(json.dumps(dataset))
    f.close()

    f = open("data/dataset/all_accords.txt", "w")
    f.write(json.dumps(ACCORDS))
    f.close()

    f = open("data/dataset/all_notes.txt", "w")
    f.write(json.dumps(NOTES))
    f.close()


def clean_graph_dataset():
    """
    Clean graph dataset
    Data format: [ID, SEX, ACCORDS, ACCORDS_VALUES, NOTES, RATING, VOTES, SIMILAR FRAGS]
    """
    f = open("data/dataset/initial_dataset.json")
    dataset = json.load(f)
    f.close()

    f = open("data/dataset/all_accords.txt")
    ACCORDS = json.load(f)
    f.close()

    f = open("data/dataset/all_notes.txt")
    NOTES = json.load(f)
    f.close()
 
    # Replace similar frags, notes and accords with their corresponding IDs
    for data in dataset:
        sim_frag_ids = []
        for sim_frag in data[9]:
            sim_frag_id = find_frag_id(sim_frag['Model'], sim_frag['Company'])
            if sim_frag_id != -1:
                sim_frag_ids.append(sim_frag_id)
        data[9] = list(dict.fromkeys(sim_frag_ids))

        accord_ids = []
        for accord in data[4]:
            accord_ids.append(ACCORDS.index(accord))
        data[4] = accord_ids

        # Normalize accord values
        data[5] = [d/sum(data[5]) for d in data[5]]

        note_ids = []
        for note in data[6]:
            note_ids.append(NOTES.index(note))
        data[6] = note_ids

    # Remove name and designer
    id_dataset = {}
    for data in dataset:
        id_dataset[data[0]] = [data[1], data[2]]
        del data[2]
        del data[1]

    f = open("data/dataset/dataset.json", "w")
    f.write(json.dumps(dataset))
    f.close()

    f = open("data/dataset/id_dataset.json", "w")
    f.write(json.dumps(id_dataset))
    f.close()


def clean_dataset():
    f = open("data/dataset/dataset.json")
    dataset = json.load(f)
    f.close()
    
    dataset_clean = []
    removed_ids = []
    # Remove data with less than 100 votes
    for data in dataset:
        if data[6] > 100:
            dataset_clean.append(data)
        else:
            removed_ids.append(data[0])
    # Remove ids of cleaned data
    for data in dataset_clean:
        data[7] = list(set(data[7])-set(removed_ids))

    N = len(dataset_clean)
    A = np.zeros((N, N))
    for i, data in enumerate(dataset_clean):
        for sim in data[7]:
            A[i, find_index_from_id(sim, dataset_clean)] = 1
    A2 = np.matmul(A, A)
    A3 = np.matmul(A2, A)
    d3 = np.sum(A3, axis=1)

    dataset_clean2 = []
    removed_ids2 = []
    # Remove unconnected nodes
    for i, data in enumerate(dataset_clean):
        if d3[i] > 300:
            dataset_clean2.append(data)
        else:
            removed_ids2.append(data[0])
    # Remove from graph completely
    for data in dataset_clean2:
        data[7] = list(set(data[7])-set(removed_ids2))

    f = open("data/dataset/dataset_clean.json", "w")
    f.write(json.dumps(dataset_clean2))
    f.close()


def create_notes_dataset():
    f = open("data/dataset/dataset_clean.json")
    dataset = json.load(f)
    f.close()

    f = open("data/dataset/all_notes.txt")
    all_notes = json.load(f)
    f.close()

    d = len(all_notes)

    data_dict = {}
    for data in dataset:
        id, sex, accords, accords_values, notes, rating, votes, sims = data
        notes_ohe = [0]*d
        for note in notes:
            notes_ohe[note] = 1
        data_dict[id] = [notes_ohe, sims]

    f = open("data/dataset/notes_dataset.json", "w")
    f.write(json.dumps(data_dict))
    f.close()




# print("Creating graph data")
# create_initial_graph_dataset()

# print("Cleaning graph data")
# clean_graph_dataset()

# print("Removing redundant data")
# clean_dataset()

print("Create notes dataset")
create_notes_dataset()
