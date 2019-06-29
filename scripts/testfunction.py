
import nltk
import json
import pickle
def generate_index():
    with open('../data/layer1.json', 'r') as f:
        layer1 = json.load(f)
    index_from_id = {}
    index_from_title = {}
    for i in range(len(layer1)):
        index_from_id[layer1[i]["id"]] = layer1[i]
        index_from_title[layer1[i]["title"]] = layer1[i]


    with open('../data/index_from_id.json', 'w') as f:
        json.dump(index_from_id, f)

    with open('../data/index_from_title.json', 'w') as f:
        json.dump(index_from_title, f)

def test_class1m():
    with open('../data/classes1M.pkl', 'rb') as f:
        class_dict = pickle.load(f)
        classindex = pickle.load(f)
    for i,key in enumerate(classindex):
        #if "cupcake" in classindex[key]:

        print (i,classindex[key])

test_class1m()