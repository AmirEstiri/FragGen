import json
import os


def find_frag_id(frag_name, designer_name):
    """
    Find the id for a fragrance using its name and designer
    """

    # TODO flexible search (maybe use regex) 

    f = open("data/dataset/initial_dataset.json")
    dataset = json.load(f)
    f.close()

    for data in dataset:
        if data[1] == frag_name and data[2] == designer_name:
            return data[0]

    return -1

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
        
        if frag_data['sex'] == "for men":
            sex = 0
        if frag_data['sex'] == "for women":
            sex = 1
        else:
            sex = 2
        
        accords = frag_data['accords']
        for accord in accords.keys():
            if accord not in ACCORDS:
                ACCORDS.append(accord)
        
        notes = frag_data['notes']['all'] + frag_data['notes']['top'] + frag_data['notes']['middle'] + frag_data['notes']['base']
        for note in notes:
            if note not in NOTES:
                NOTES.append(note)
        
        rating = frag_data['rating']
        total_votes = frag_data['total_votes']
        similar = frag_data['similar_fragrances']

        dataset.append([id, name, designer, sex, accords, notes, rating, total_votes, similar])

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
        for sim_frag in data[8]:
            sim_frag_id = find_frag_id(sim_frag['Model'], sim_frag['Company'])
            if sim_frag_id != -1:
                print(sim_frag_id)
                sim_frag_ids.append(sim_frag_id)
        data[8] = sim_frag_ids

        accord_ids = []
        for accord in data[4]:
            accord_ids.append(ACCORDS.index(accord))
        data[4] = accord_ids

        note_ids = []
        for note in data[5]:
            note_ids.append(NOTES.index(note))
        data[5] = note_ids

    f = open("data/dataset/dataset.json", "w")
    f.write(json.dumps(dataset))
    f.close()


create_initial_graph_dataset()
clean_graph_dataset()