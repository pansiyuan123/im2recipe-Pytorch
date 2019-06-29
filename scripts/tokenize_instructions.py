import json
import numpy as np
import pickle
import sys
from tqdm import *
import time

def readfile(filename):
    with open(filename,'r') as f:
        lines = []
        for line in f.readlines():
            lines.append(line.rstrip())
    return lines

def tok(text,ts=False):

    '''
    Usage: tokenized_text = tok(text,token_list)
    If token list is not provided default one will be used instead.
    '''

    if not ts:
        ts = [',','.',';','(',')','?','!','&','%',':','*','"']

    for t in ts:
        text = text.replace(t,' ' + t + ' ')
    return text

if __name__ == "__main__":

    '''
    Generate tokenized text for w2v training
    Words separated with ' '
    Different instructions separated with \t
    Different recipes separated with \n
    '''

    try:
        partition = str(sys.argv[1])
    except:
        partition = ''

    dets = json.load(open('../data/recipe1M/det_ingrs.json','r'))
    layer1 = json.load(open('../data/recipe1M/layer1.json','r'))

    idx2ind = {}
    ingrs = []
    for i,entry in enumerate(dets):
        idx2ind[entry['id']] = i


    t = time.time()
    print ("Saving tokenized here:", 'files/tokenized_instructions_'+partition+'.txt')
    f = open('files/tokenized_instructions_'+partition+'.txt','w')
    for i,entry in tqdm(enumerate(layer1)):
        '''
        if entry['id'] in dups:
            continue
        '''
        if not partition=='' and not partition == entry['partition']:
            continue
        instrs = entry['instructions']

        allinstrs = ''
        for instr in instrs:
            instr =  instr['text']
            allinstrs+=instr + '\t'

        # find corresponding set of detected ingredients
        det_ingrs = dets[idx2ind[entry['id']]]['ingredients']
        valid = dets[idx2ind[entry['id']]]['valid']

        for j,det_ingr in enumerate(det_ingrs):
            # if detected ingredient matches ingredient text,
            # means it did not work. We skip
            if not valid[j]:
                continue
            # underscore ingredient

            det_ingr_undrs = det_ingr['text'].replace(' ','_')
            ingrs.append(det_ingr_undrs)
            allinstrs = allinstrs.replace(det_ingr['text'],det_ingr_undrs)

        f.write(allinstrs + '\n')

    f.close()
    print (time.time() - t, 'seconds.')
    print ("Number of unique ingredients",len(np.unique(ingrs)))
    f = open('files/tokenized_instructions_'+partition+'.txt','r')
    text = f.read()
    text = tok(text)
    f.close()

    f = open('files/tokenized_instructions_'+partition+'.txt','w')
    f.write(text)
    f.close()
    #菜谱中全部是步骤
    '''
    Preheat the oven to 350 F .  Butter or oil an 8-inch baking dish . 	Cook the penne 2 minutes less than package directions . 	 ( It will finish cooking in the oven .  ) 	Rinse the pasta in cold water and set aside . 	Combine the cooked pasta and the sauce in a medium bowl and mix carefully but thoroughly . 	Scrape the pasta into the prepared baking dish . 	Sprinkle the top with the cheeses and then the chili powder . 	Bake ,  uncovered ,  for 20 minutes . 	Let the mac and cheese sit for 5 minutes before serving . 	Melt the butter in a heavy-bottomed saucepan over medium heat and whisk in the flour . 	Continue whisking and cooking for 2 minutes . 	Slowly add the milk ,  whisking constantly . 	Cook until the sauce thickens ,  about 10 minutes ,  stirring frequently . 	Remove from the heat . 	Add the cheeses ,  salt ,  chili powder ,  and garlic_powder . 	Stir until the cheese is melted and all ingredients are incorporated ,  about 3 minutes . 	Use immediately ,  or refrigerate for up to 3 days . 	This sauce reheats nicely on the stove in a saucepan over low heat . 	Stir frequently so the sauce doesnt scorch . 	This recipe can be assembled before baking and frozen for up to 3 monthsjust be sure to use a freezer-to-oven pan and increase the baking time to 50 minutes . 	One-half teaspoon of chipotle chili powder makes a spicy mac ,  so make sure your family and friends can handle it ! 	The proportion of pasta to cheese_sauce is crucial to the success of the dish . 	It will look like a lot of sauce for the pasta ,  but some of the liquid will be absorbed . 	
Cook macaroni according to package directions ;  drain well . 	Cold . 	Combine macaroni ,  cheese cubes ,  celery ,  green pepper and pimento . 	Blend together mayonnaise or possibly salad dressing ,  vinegar ,  salt and dill weed ;  add in to macaroni mix . 	Toss lightly . 	Cover and refrigeratewell . 	Serve salad in lettuce lined bowl if you like . 	Makes 6 servings . 	
Add the tomatoes to a food processor with a pinch of salt and puree until smooth . 	Combine the onions ,  bell peppers and cucumbers with the tomato puree in a large bowl . 	Chill at least 1 hour . 	Drizzle with olive_oil ,  garnish with chopped basil and serve . 	
Dissolve Jello in boiling_water . 	Allow to cool to room temp . 	Whisk in Cool_Whip . 	Fold in watermelon . 	Spoon into crust . 	Chill for 2-3 hours or overnight . 	Yum ! 	
In a large skillet ,  toast the coconut over medium heat ,  until golden and crisp ;  set aside . 	Brown ground beef and garlic in the same skillet ;  drain well . 	Add salt ,  pepper lemon_juice and soy_sauce . 	In a small bowl combine the cornstarch with reserved pineapple and mandarin orange liquids ;  stir well until smooth then add to ground beef and cook over medium heat for 5 mins ,  stirring constantly ,  until mixture is thickened . 	Stir in the pineapple and mandarin_oranges ;  cook 2-3 mins ,  or until thoroughly heated . 	Serve over noodles or rice ,  and sprinkle with more toasted coconut and cashew_nuts . 	
Pierce the skin of the chicken with a fork or knife . 	Sprinkle with kombu tea evenly on both sides of the chicken ,  about 1 teaspoon per chicken thigh . 	Brown the skin side of the chicken first over high heat until golden brown . 	Sprinkle some pepper on the meat just before flipping over . 	Then brown the other side until golden brown . 	
Put ingredients in a buttered 9 x 12 x 2-inch pan in even layers in the order that they are given - DO NOT MIX . 	Bake in a 350 oven for 1 hour . 	
Layer all ingredients in a serving dish . 
    '''