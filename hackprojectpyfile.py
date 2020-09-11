#import all libraries required
""" 
pandas - for dataframe,
numpy - creating array and manipulation
nltk.tokenize for tokenization 
math - for log10 
string for punctuation 
matplotlib - for ploting graph
sklearn = to calcalulate accuracy of prediction or classifier
"""

import pandas as pd
from nltk.tokenize import word_tokenize
import numpy as np
import math
import string
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

"""
read the datafile as csv format
"""

dataset = pd.read_csv("hns_2018_2019.csv")

#split dataset according as 2018 year as training dataset
# nad 2019 year as testing dataset

training_set=dataset.loc[dataset['year']==2018,['Title','year','Post Type']]

testing_set=dataset.loc[dataset['year']==2019,['Title','year','Post Type']]



#after that we will use training dataset to create model 
"""
steps for creating model 

    1. categorised title according to post type so we have three category story, ask_hn,show_hn
    2. for each category tokenize each to its corresponding category and also calculate frequency of each word
    
    after step 1 and 2 we have all words with category and frequncy in its coresponding category
    
    now we will make vocabulary by merging all words and calculating frequncy of each word
    
    after that we have vocabulary as 
    
    word          story     ask-hn       show-hn
    
    terminal        12        5             0
    the             232       234          23


    3. after getting vocabualry we will compute p(wi/post_type) for each word in vocabulary 
    
        and this data will merge with vocabulary called model and save as 2018-model.txt
    
"""
#we have three category story ask-hn and show-hn so all title wiil divide in three coresponding category

story=training_set.loc[training_set['Post Type']=='story',['Title']]
ask_hn=training_set.loc[training_set['Post Type']=='ask_hn',['Title']]
show_hn=training_set.loc[training_set['Post Type']=='show_hn',['Title']]

#tokenize each title of story and store each word in vb_of_story 
#and its frequency will be stored in fr_of_sotry

vb_of_story=[]
fr_of_story=[]

for word in story['Title']:
    word=word.lower()
    word=word_tokenize(word)
    for token in word:
        if token in string.punctuation:
            continue
        if (token not in vb_of_story):
            vb_of_story.append(token)
            fr_of_story.append(1)
        else:
            fr_of_story[vb_of_story.index(token)]+=1


#same will perform with ask_hn as well as show_Hn

vb_of_show_hn=[]
fr_of_show_hn=[]

for word in show_hn['Title']:
    word=word.lower()
    word=word_tokenize(word)
    for token in word:
        if token in string.punctuation:
            continue
        if token not in vb_of_show_hn:
            vb_of_show_hn.append(token)
            fr_of_show_hn.append(1)
        else:
            fr_of_show_hn[vb_of_show_hn.index(token)]+=1

vb_of_ask_hn=[]
fr_of_ask_hn=[]

for word in ask_hn['Title']:
    word=word.lower()
    word=word_tokenize(word)
    for token in word:
        if token in string.punctuation:
            continue

        if token not in vb_of_ask_hn:
            vb_of_ask_hn.append(token)
            fr_of_ask_hn.append(1)
        else:
            fr_of_ask_hn[vb_of_ask_hn.index(token)]+=1

#now we calculate each word's frequncy in all categpry and store it in vb 
# and its frequency into fr        

vb=[]
fr=[]

for word in training_set['Title']:
    word=word.lower()
    word=word_tokenize(word)
    for token in word:
        if token in string.punctuation:
            continue
        if token not in vb:
            vb.append(token)
            fr.append(1)
        else:
            fr[vb.index(token)]+=1



story=np.zeros(len(vb),dtype=int)
ask_hn=np.zeros(len(vb),dtype=int)
show_hn=np.zeros(len(vb),dtype=int)
poll=np.zeros(len(vb),dtype=int)


for word in vb:
    if word in vb_of_story:
        story[vb.index(word)]+= fr_of_story[vb_of_story.index(word)]
    if word in vb_of_ask_hn:
        ask_hn[vb.index(word)]+=fr_of_ask_hn[vb_of_ask_hn.index(word)]
    if word in vb_of_show_hn:
        show_hn[vb.index(word)]+=fr_of_show_hn[vb_of_show_hn.index(word)]
  
#after calcaulating each word's frequency into all post_type we will put all this information at one place called vocabualry using pandas dataframe
        
vocabulary = pd.DataFrame({
        "word":vb,
        "story":story,
        "ask_hn":ask_hn,
        "show_hn":show_hn,
        "poll":poll,
        "frequency":fr},index=vb)
    
    
#now we will calculate post_type probability for each post
   """
   as total number of post is 5000 
   p(story)= number of story/(total)
   p(ask_hn)=number of ask_Hn/total
   p(show_hn)=number of show_hn/total
   
   """
post_type = training_set['Post Type'].value_counts()
prob_set=pd.DataFrame({
        "story":post_type[0]/5000,
        "ask_hn":post_type[1]/5000,
        "show_hn":post_type[2]/5000},index=[1])


    
# we are defining a function named sm)cnd_pro which calcualtes 
#    smoothed conditional probability for given word and post
#   it takes wi = number of in the post 
#           post = total no of word in post
# and round it to 6 decimal    
    
    
def sm_cnd_pro(wi,post):
    smooth=0.5
    wi+=smooth
    return round((wi/post),6)

#for each word word in vocabulary we calculate smoothed conditional probabli;ity
    # for each post

cnd_prob_story=[]
cnd_prob_ask_hn=[]
cnd_prob_show_hn=[]
cnd_prob_poll=[]

for word in vb:
    index=vb.index(word)
    cnd_prob_story.append(sm_cnd_pro(story[index],post_type[0]))

for word in vb:
    index=vb.index(word)
    cnd_prob_ask_hn.append(sm_cnd_pro(ask_hn[index],post_type[1]))

for word in vb:
    index=vb.index(word)
    cnd_prob_show_hn.append(sm_cnd_pro(show_hn[index],post_type[2]))
    
for word in vb:
    cnd_prob_poll.append(-99)
    
    
# after calculating smoothed conditional probablility we meerge this with vocabulary

vocabulary["cnd_prob_story"]=cnd_prob_story
vocabulary["cnd_prob_ask_hn"]=cnd_prob_ask_hn
vocabulary["cnd_prob_show_hn"]=cnd_prob_show_hn
vocabulary["cnd_prob_poll"]=cnd_prob_poll


#now we have model 
# write vocabulary to into text file named vocabulary.txt

file=open('vocabulary.txt','w+',encoding="utf-8")
vocabulary=vocabulary.sort_values('word')
for word in vocabulary.index:
    file.write(word)
    file.write("\n")
file.close()

#funcition to write model into text file


def write_model(vocabulary,name):
    file=open(name,'w+',encoding="utf-8")
    n=1
    for word in vocabulary.index:
        curr=vocabulary.loc[word]
        
        file.write(str(n)+"  ")
        file.write(str(word)+"  ")
        file.write(str(int(curr['story']))+"  ")
        file.write(str(curr['cnd_prob_story'])+'  ')
        file.write(str(int(curr['ask_hn']))+'  ')
        file.write(str(curr['cnd_prob_ask_hn'])+'  ')
        file.write(str(int(curr['show_hn']))+'  ')
        file.write(str(curr['cnd_prob_show_hn'])+'  ')
        file.write(str(int(curr['poll']))+'  ')
        file.write(str(curr['cnd_prob_poll'])+'\n')
        n+=1
    file.close()

#write model to text file 
write_model(vocabulary,'model-2018.txt')


#now we will make classifier which calcaulates probability of each post with aords
# it returns reulsts which contains actual post type with prediced post_type
def classifier(vocabulary):    
    pre_class={0:'story',1:'ask_hn',2:'show_hn'}
    
    score=[]
    predicted=[]
    for post in testing_set['Title']:
        post = post.lower()
        post=word_tokenize(post)
        class_prob=[round(math.log10(prb),4) for prb in prob_set.iloc[0]]
        for word in post:
            if word in string.punctuation:
                continue
            if word in vocabulary.index:
                row=vocabulary[vocabulary.index==word]
                class_prob[0]+=math.log10(row['cnd_prob_story'])
                class_prob[1]+=math.log10(row['cnd_prob_ask_hn'])
                class_prob[2]+=math.log10(row['cnd_prob_show_hn'])
        score.append(class_prob)
        predicted.append(pre_class[class_prob.index(max(class_prob))])
        
    
    result =pd.DataFrame({
            'Title':testing_set['Title'],
            'actual':testing_set['Post Type'],
            'predicted':predicted,
            'score':score})
    return result
        

#run the classifier with giving model and sotre it in result


result=classifier(vocabulary)

# check the result by accuracy and confuasion matrix and classification repsot
confusion_matrix(result['actual'],result['predicted'])
accuracy_score(result['actual'],result['predicted'])
classification_report(result['actual'],result['predicted'])

# here baseline accuracy is 42.6% 


#  write_result is function which will write result ot text file

def write_result(result,name):
    file=open(name,'w+',encoding='utf8')
    n=1
    for row in result.index:
        doc=result.loc[row]
        file.write(str(n)+"  ")
        file.write(str(doc['Title'][0:15] )+ "..  ")
        file.write(str(doc['predicted'])+"  ")
        file.write(str(doc['score'][0])+"  ")
        file.write(str(doc['score'][1])+"  ")
        file.write(str(doc['score'][2])+"  ")
        file.write(str(doc['actual'])+"  ")
        file.write(str('right' if doc['predicted']==doc['actual'] else 'wrong')+"\n")
        n+=1
    file.close()
write_result(result,'baseline-result.txt')

#task 3.1
# store all words in stopwords.txt and then drop all stopwords from vocabualry or model

file=open('stopwords.txt','r+',encoding='utf-8')
stopwords=[word.replace('\n','') for word in file if word.replace('\n','') in vocabulary.index]
file.close()
stop_words_vcb=vocabulary.drop(stopwords)

# run the classifier 
result=classifier(stop_words_vcb)

# after succesfull write vocabulary into textfile named stopw_words.txt and 
# result to stop_words_result.txt

write_model(stop_words_vcb,' stopword-model.txt')
write_result(result,'stopword-result.txt')

# calculate the accuracy
accuracy_score(result['actual'],result['predicted'])

# accuracy is 45%

#task 3.2

#drop all the words as task 3.2 told
dropped=[word for word in vocabulary.index if len(word)<=2 or len(word)>=9]
vocabulary2=vocabulary.drop(dropped)


# run the classifier and write to wordlength_model.text
# and wordlength_results.txt
result=classifer(vocabulary2)
write_model(stop_words_vcb,' wordlength-model.txt')
write_result(result,'wordlength-result.txt')

# chech the result
confusion_matrix(result['actual'],result['predicted'])
accuracy_score(result['actual'],result['predicted'])
classification_report(result['actual'],result['predicted'])

# accuracy is 72%

#task 1.3

#gradually drop words which have frequency less than 5, 10, 15 and 20 and calculate word left with performance

words_left=[]
performance=[]
ran=[5,10,15,20]
for r in ran:
    dropped=[word for word in vocabulary2.index if vocabulary2.loc[word]['frequency']<=r]
    vocabulary2=vocabulary2.drop(dropped)
    result=classifier(vocabulary2)
    words_left.append(len(vocabulary2))
    performance.append(accuracy_score(result['actual'],result['predicted']))


#gradually drop the top 5%. 10% , 15% , 20% and 25% frequncy word and calculate
    # and calcualte words_left vs performance 
    
ran=[5,10,15,20,25]
for r in ran:
    vocabulary2=vocabulary2.sort_values('frequency', ascending=False)
    dropped=[word for word in vocabulary2.head(r).index]
    vocabulary3=vocabulary2.drop(dropped)
    result=test_data(vocabulary3)
    words_left.append(len(vocabulary3))
    #confusion_matrix(result['actual'],result['predicted'])
    performance.append(accuracy_score(result['actual'],result['predicted']))
#classification_report(result['actual'],result['predicted'])

plt.scatter(words_left,performance)
plt.show()




