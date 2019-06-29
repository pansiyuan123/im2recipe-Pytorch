import json
import re

import copy
import pickle
from proc import *
from params import get_parser
import utils


parser = get_parser()
params = parser.parse_args()

create = params.create_bigrams


print('Loading dataset.')
DATASET = params.dataset
dataset = utils.Layer.merge([utils.Layer.L1, utils.Layer.L2, utils.Layer.INGRS],DATASET)

'''
for i in range(len(dataset)):
    print (dataset[i])
    if i>100:
        break
    
    {
     'ingredients':
         [
         {'text': 'creamy peanut butter'}, 
         {'text': 'Crisco shortening'}, 
         {'text': 'light brown sugar'}, 
         {'text': 'milk'}, 
         {'text': 'vanilla'}, 
         {'text': 'egg'}, 
         {'text': 'flour'}, 
         {'text': 'salt'}, 
         {'text': 'baking soda'}
         ], 
     'url': 'http://www.food.com/recipe/irresistible-peanut-butter-cookies-210312', 
     'partition': 'val', 
     'title': 'Irresistible Peanut Butter Cookies', 
     'id': '0006354bc3', 
     'instructions': 
         [
         {'text': 'Combine peanut butter, Crisco, brown sugar, milk, and vanilla in a large bowl.'}, 
         {'text': 'Beat at medium speed until well blended.'}, 
         {'text': 'Add egg and beat just until blended.'}, 
         {'text': 'Mix in the flour, salt, and baking soda at low speed and mix just till blended.'}, 
         {'text': 'Drop by heaping teaspoonfuls about 2" apart onto an ungreased sheet.'}, 
         {'text': 'Flatten slightly in crisscross pattern with tines of a floured fork.'}, 
         {'text': 'Bake at 375 degrees for 7-8 minutes or until set and just beginning to brown.'}, 
         {'text': 'Cool for 2 minutes on the sheet before removing.'}
         ], 
     'images': 
         [
         {
         'id': '7027a5c4f9.jpg', 
         'url': 'http://img.sndimg.com/food/image/upload/w_512,h_512,c_fit,fl_progressive,q_95/v1/img/recipes/21/03/12/pics2duVm.jpg'
         }, 
         {'id': 'fff3e1751a.jpg', 
         'url': 'http://img.sndimg.com/food/image/upload/w_512,h_512,c_fit,fl_progressive,q_95/v1/img/recipes/21/03/12/picrPS1es.jpg'
         }
         ], 
     'valid': [True, True, True, True, True, True, True, True, True]
     }
    '''

if create:
    print("Creating bigrams...")
    titles = []
    for i in range(len(dataset)):
        title = dataset[i]['title']
        if dataset[i]['partition'] == 'train':
            titles.append(title)
    fileinst = open('../data/titles' + params.suffix + '.txt','w')
    for t in titles:
        fileinst.write( t + " ")

    '''
    所有的菜名
    Strawberry Rhubarb Dump Cake Yogurt Parfaits Zucchini Nut Bread
    '''

    fileinst.close()

    import nltk
    from nltk.corpus import stopwords
    f = open('../data/titles' +params.suffix+'.txt')
    raw = f.read()
    tokens = nltk.word_tokenize(raw)
    tokens = [i.lower() for i in tokens]
    tokens = [i for i in tokens if i not in stopwords.words('english')]
    #Create your bigrams
    bgs = nltk.bigrams(tokens)
    #compute frequency distribution for all the bigrams in the text
    fdist = nltk.FreqDist(bgs)
    '''
    for i,key in enumerate(fdist):
        print (i,key,fdist[key])
        if i>100:
            break
    
    #each two words a tuple,后面是词频 18 （“fuck1”，“fuck2”） 206
    '''
    pickle.dump(fdist, open('../data/bigrams' + params.suffix + '.pkl', 'wb'))

else:
    N = 2000
    MAX_CLASSES = 1000
    MIN_SAMPLES = params.tsamples #20
    n_class = 1
    ind2class = {}
    class_dict = {}

    fbd_chars = [",", "&", "(", ")", "'", "'s", "!", "?", "%", "*", ".",
                 "free", "slow", "low", "old", "easy", "super", "best", "-", "fresh",
                 "ever", "fast", "quick", "fat", "ww", "n'", "'n", "n", "make", "con",
                 "e", "minute", "minutes", "portabella", "de", "of", "chef", "lo",
                 "rachael", "poor", "man", "ii", "i", "year", "new", "style"]

    print ("loading ingr vocab")
    with open(params.vocab) as f_vocab:
        ingr_vocab = {w.rstrip(): i+2 for i, w in enumerate(f_vocab)} # +1 for lua
        ingr_vocab['</i>'] = 1

    '''
    用前面词袋vocab.txt生成，一个单词，后面是序号+2
        for i,key in enumerate(ingr_vocab):
        print (i,key,ingr_vocab[key])
        if i>100:
            break
    16 into 18
    17 over 19
    18 heat 20
    19 bowl 21
    '''


    ningrs_list = []
    for i, entry in enumerate(dataset):
        ingr_detections = detect_ingrs(entry, ingr_vocab)
        if i<10:
            print (ingr_detections)
        ningrs = len(ingr_detections)
        ningrs_list.append(ningrs)
        '''
        查找每一个菜谱的ingredients中有几个单词可以在词袋中找到
        ingr_detections是一个list，中间是这个词在词袋中的编号
        这个函数写的就是一个智障
        '''

    #print (ningrs_list[:10])
    fdist = pickle.load(open('../data/bigrams' + params.suffix + '.pkl', 'rb')) #tuple+词频 ("fuck1","fuck2") 206
    Nmost = fdist.most_common(N)

    queries = []
    #对每一个高频词汇在菜谱中进行查询
    for oc in Nmost:

        counts = {'train': 0, 'val': 0, 'test': 0}

        if oc[0][0] in fbd_chars or oc[0][1] in fbd_chars:
            continue

        query = oc[0][0] + ' ' + oc[0][1]
        queries.append(query)
        matching_ids = []

        for i, entry in enumerate(dataset):

            ninstrs = len(entry['instructions'])#步骤的长度
            imgs = entry.get('images')
            '''
                    [
                     {
                     'id': '7027a5c4f9.jpg', 
                     'url': 'http://img.sndimg.com/food/image/upload/w_512,h_512,c_fit,fl_progressive,q_95/v1/img/recipes/21/03/12/pics2duVm.jpg'
                     }, 
                     {'id': 'fff3e1751a.jpg', 
                     'url': 'http://img.sndimg.com/food/image/upload/w_512,h_512,c_fit,fl_progressive,q_95/v1/img/recipes/21/03/12/picrPS1es.jpg'
                     }
                     ]
            '''
            ningrs = ningrs_list[i] #这个菜谱有几个单词能被找到
            title = entry['title'].lower()
            id = entry['id']
            #以上是对每一个菜谱的信息进行解析

            #一个菜谱的title可能会在多个高词频的tuple中被找到
            if query in title and ninstrs < params.maxlen and imgs and ningrs < params.maxlen and ningrs is not 0:
                # if match, add class to id
                # we only add if previous class was background
                # or if there is no class for the id
                if id in class_dict:
                    if class_dict[id] == 0:
                        class_dict[id] = n_class #这一个菜谱是第几类
                        counts[dataset[i]['partition']] += 1
                        matching_ids.append(id)
                else:
                    class_dict[id] = n_class
                    counts[dataset[i]['partition']] += 1
                    matching_ids.append(id)

            else:  # if there's no match
                if not id in class_dict:  # add background class unless not empty
                    class_dict[id] = 0  # background class 这一类没有东西

        #训练测试验证都要有
        if counts['train'] > MIN_SAMPLES and counts['val'] > 0 and counts['test'] > 0:
            ind2class[n_class] = query
            print(n_class, query, counts)
            n_class += 1
        else:
            for id in matching_ids:  # reset classes to background
                class_dict[id] = 0

        if n_class > MAX_CLASSES:
            break

    # get food101 categories (if not present)
    food101 = []
    with open(params.f101_cats, 'r') as f_classes:
        for l in f_classes:
            cls = l.lower().rstrip().replace('_', ' ')
            if cls not in queries:
                food101.append(cls)

    for query in food101:
        counts = {'train': 0, 'val': 0, 'test': 0}
        matching_ids = []
        for i, entry in enumerate(dataset):

            ninstrs = len(entry['instructions'])
            imgs = entry.get('images')
            ningrs = ningrs_list[i]
            title = entry['title'].lower()
            id = entry['id']

            if query in title and ninstrs < params.maxlen and imgs and ningrs < params.maxlen and ningrs is not 0:  # if match, add class to id
                # we only add if previous class was background
                # or if there is no class for the id
                if id in class_dict:
                    if class_dict[id] == 0:
                        class_dict[id] = n_class
                        counts[dataset[i]['partition']] += 1
                        matching_ids.append(id)
                else:
                    class_dict[id] = n_class
                    counts[dataset[i]['partition']] += 1
                    matching_ids.append(id)

            else:  # if there's no match
                if not id in class_dict:  # add background class unless not empty
                    class_dict[id] = 0  # background class

        if counts['train'] > MIN_SAMPLES and counts['val'] > 0 and counts['test'] > 0:
            ind2class[n_class] = query
            print(n_class, query, counts)
            n_class += 1
        else:
            for id in matching_ids:  # reset classes to background
                class_dict[id] = 0

    ind2class[0] = 'background'
    print(len(ind2class))
    with open('../data/classes' + params.suffix + '.pkl', 'wb') as f:
        pickle.dump(class_dict, f)
        pickle.dump(ind2class, f)
    #clss1m.pkl
    #classdict 是这一个菜谱的属于哪一类的编号
    #ind2class是这一个编号到对应的菜名(总共只有1000类)
