
# coding: utf-8

# In[20]:

import sys

from itertools import chain, combinations
from collections import defaultdict
from optparse import OptionParser


def subsets(arr):
    """ Returns non empty subsets of arr"""
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])


def returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet):
        """calculates the support for items in the itemSet and returns a subset
       of the itemSet each of whose elements satisfies the minimum support"""
        _itemSet = set()
        localSet = defaultdict(int)

        for item in itemSet:
                for transaction in transactionList:
                        if item.issubset(transaction):
                                freqSet[item] += 1
                                localSet[item] += 1

        for item, count in localSet.items():
                support = float(count)/len(transactionList)

                if support >= minSupport:
                        _itemSet.add(item)

        return _itemSet


def joinSet(itemSet, length):
        """Join a set with itself and returns the n-element itemsets"""
        return set([i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length])


def getItemSetTransactionList(data_iterator):
    transactionList = list()
    itemSet = set()
    for record in data_iterator:
        transaction = frozenset(record)
        transactionList.append(transaction)
        for item in transaction:
            itemSet.add(frozenset([item]))              # Generate 1-itemSets
    return itemSet, transactionList


def runApriori(data_iter, minSupport, minConfidence):
    """
    run the apriori algorithm. data_iter is a record iterator
    Return both:
     - items (tuple, support)
     - rules ((pretuple, posttuple), confidence)
    """
    itemSet, transactionList = getItemSetTransactionList(data_iter)

    freqSet = defaultdict(int)
    largeSet = dict()
    # Global dictionary which stores (key=n-itemSets,value=support)
    # which satisfy minSupport

    assocRules = dict()
    # Dictionary which stores Association Rules

    oneCSet = returnItemsWithMinSupport(itemSet,
                                        transactionList,
                                        minSupport,
                                        freqSet)

    currentLSet = oneCSet
    k = 2
    while(currentLSet != set([])):
        largeSet[k-1] = currentLSet
        currentLSet = joinSet(currentLSet, k)
        currentCSet = returnItemsWithMinSupport(currentLSet,
                                                transactionList,
                                                minSupport,
                                                freqSet)
        currentLSet = currentCSet
        k = k + 1

    def getSupport(item):
            """local function which Returns the support of an item"""
            return float(freqSet[item])/len(transactionList)

    toRetItems = []
    for key, value in largeSet.items():
        toRetItems.extend([(tuple(item), getSupport(item))
                           for item in value])

    toRetRules = []
    for key, value in largeSet.items()[1:]:
        for item in value:
            _subsets = map(frozenset, [x for x in subsets(item)])
            for element in _subsets:
                remain = item.difference(element)
                if len(remain) > 0:
                    confidence_lift = []
                    item_A=getSupport(element)
                    item_B=getSupport(remain)
                    confidence_lift.append(getSupport(item)/item_A)#conf - 0
                    if confidence_lift[0] >= minConfidence:
                        confidence_lift.append(getSupport(item)/(item_A*item_B))#lift - 1
                        if  item_A>item_B:
                            confidence_lift.append(item_B)#min(minsup) - 2
                        else:
                            confidence_lift.append(item_A)
                        toRetRules.append(((tuple(element), tuple(remain)),
                                           confidence_lift))
    return toRetItems, toRetRules



def dataFromFile(fname):
        """Function which reads from the file and yields a generator"""
        file_iter = open(fname, 'rU')
        for line in file_iter:
                line = line.strip().rstrip(',')                         # Remove trailing comma
                record = frozenset(line.split(','))
                yield record
                
def printResults(items, rules, rs,it, idc, t_d,start_date,end_date, len_ids, action_pack): 
                                        #принимает: итемы для минсапа, правила с конф, файл для записи результата,
                                        #название текущего пака, номер айдишника для записи в словарь данных по пакам (tableau),
                                        #словарь, дата начала и конца, кол-во транзакций в периоде
    """prints the generated itemsets sorted by support and the confidence rules sorted by confidence"""
    if len(rules)!=0:
        rs.write("SUP------------SUP------------SUP:\n")
        rs.write("For time boundaries on: "+it+"\n") #показывает периоды какого пака исследуются
        for item, support in sorted(items, key=lambda (item, support): support):
            rs.write("item: %s , %.3f \n" % (str(item), support))
            #print "item: %s , %.3f" % (str(item), support)
        rs.write("\nRULES------------RULES------------RULES:\n")
        for rule, confidence in sorted(rules, key=lambda (rule, confidence): confidence):
            pre, post = rule
            idc=idc+1
            t_d[idc]=[]
            st_st="%s ==> %s"%(str(pre), str(post))
            t_d[idc].append(st_st)#rule 
            t_d[idc].append(confidence[0])#conf
            t_d[idc].append(confidence[1])#lift
            t_d[idc].append(confidence[2])#min minsup
            if "fb_" in str(pre)[:7]:
                if "fb_" in str(post)[:7]:
                    t_d[idc].append("fb") #social_network
                else:
                    t_d[idc].append("cross")
            elif "vk_" in str(pre)[:7]:
                if "vk_" in str(post)[:7]:
                    t_d[idc].append("vk") #social_network
                else:
                    t_d[idc].append("cross")
            elif "ok_" in str(pre)[:7]:
                if "ok_" in str(post)[:7]:
                    t_d[idc].append("ok") #social_network
                else:
                    t_d[idc].append("cross")
            elif "mm_" in str(pre)[:7]:
                if "mm_" in str(post)[:7]:
                    t_d[idc].append("mm") #social_network
                else:
                    t_d[idc].append("cross")
            else:
                t_d[idc].append(" ")
            date_for_period=str(date.fromtimestamp(start_date))+" to "+str(date.fromtimestamp(end_date))
            t_d[idc].append(date_for_period)#date
            t_d[idc].append(len_ids)# number of transactions in period
            
            l_s=str(pre)
            r_s=str(post)
            access_act = False
            l=l_s.strip("(").strip(")").strip(",").split(",")
            for i in range(0,len(l)):
                l[i]=l[i].strip("'")
                if l[i] in action_pack:
                    access_act=True
            r=r_s.strip("(").strip(")").strip(",").split(",")
            for i in range(0,len(r)):
                r[i]=r[i].strip("'")
                if r[i] in action_pack:
                    access_act=True
            if access_act==True:
                t_d[idc].append("y")# if rule consist action pack
            else:
                t_d[idc].append("n")
            rs.write("Rule: %s ==> %s , confidence: %.3f, lift: %.3f, min(minsup) %.3f\n" % (str(pre), str(post), confidence[0], confidence[1], confidence[2]))
        rs.write("\n-----------------------------------------\n")
        rs.write("\n")
    return idc, t_d


if __name__ == "__main__":

    optparser = OptionParser()
    optparser.add_option('-f', '--inputFile',
                         dest='input',
                         help='filename containing csv',
                         default=None)
    optparser.add_option('-s', '--minSupport',
                         dest='minS',
                         help='minimum support value',
                         default=0.15,
                         type='float')
    optparser.add_option('-c', '--minConfidence',
                         dest='minC',
                         help='minimum confidence value',
                         default=0.6,
                         type='float')

    (options, args) = optparser.parse_args()


    inFile = dataFromFile("without_bound.csv")

    minSupport = options.minS
    minConfidence = options.minC

    items, rules = runApriori(inFile, minSupport, minConfidence)

    #printResults(items, rules)


# In[16]:

import sys, errno

import csv
import sys

def id_soc_data_inf(): 
    #Создаем словарь айди-соц данные. Пока что тут 
    #лежит только страна. Ошибка выдается на нечисловой айдишник.
    
    now = datetime.now()
    # 4 - country
    id_soc_data={}
    with open('w_pinfo.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='|', quotechar='|')
        spamreader.next()
        for row in spamreader:
            try:
                if id_soc_data.has_key(int(row[0]))==False:
                    id_soc_data[int(row[0])]=[]
                    id_soc_data[int(row[0])].append(row[5])
            except ValueError as e:
                print("Error has been found") #вылез странный айдишник 2.392...Е, пока что кидаю в еррор
                now1 = datetime.now()
                print(now1-now)
                now = datetime.now()
    now1 = datetime.now()
    print (now1-now)
    return id_soc_data


# In[22]:

from datetime import datetime
from datetime import date
def write_res_to_file_name_lvl2(file_name, mnsp, mncf, t_d, action_pack, id_soc_data, id_pack_period):
    id_count=0
    now = datetime.now()
    minSupport=mnsp
    minConfidence=mncf

    count=0
    with open(file_name, 'w') as rs:
        for it,time_st_end in action_pack.items():
            start_time=time_st_end[0]
            end_time=time_st_end[1]
            for bound in range(0,len(start_time)):
                ids={}
                # 3) если пак акционный, то забиваем данные для составления ассоциативных правил
                
                for id_id in id_pack_period:
                    temp=[]
                    for packs in id_pack_period[id_id]:
                        acs=False
                        for iterator in range(0,len(packs)-1): # проверяем покупался ли пак данным пользователем в данный промежуток времени
                            if (packs[iterator+1]>=start_time[bound] and packs[iterator+1]<=end_time[bound]):
                                acs=True
                        if acs==True:
                            temp.append(packs[0])
                    if len(temp)!=0:
                        ids[id_id]=[]
                        for item in  temp:
                            ids[id_id].append(item)
                
                with open('file_for_transactions.csv', 'w') as csf:
                    for key, value in ids.items():
                        st=""
                        for i in range(0,len(value)):
                            st=st+value[i]+','
                        if id_soc_data.has_key(key):
                            if id_soc_data[key][0]!=" ":    
                                st=st+id_soc_data[key][0]+","
                        csf.write(st+'\n')
                #now1 = datetime.now()
                #print "файлы: ","  ",(now1-now)

                inFile = dataFromFile("file_for_transactions.csv")
                items, rules = runApriori(inFile, minSupport, minConfidence)    
                id_count, t_d=printResults(items, rules, rs,it, id_count, t_d, start_time[bound],end_time[bound],len(ids), action_pack)
                #print len(ids),"-",
                


    # 4) время работы алгоритма
    now1 = datetime.now()
    print (now1-now)
    return t_d

def id_pack_period_lvl2(social_ntwrk): # на вход подаем какая social network исследуется
    social_network=social_ntwrk
    id_pack_period={}
    now = datetime.now()
    with open('trans.csv', 'rb') as csvfile:
        spam = csv.reader(csvfile, delimiter='|', quotechar='|')
        spam.next()
        for row in spam:
            if row[1]==social_network:
                if id_pack_period.has_key(int(row[0]))==True:
                    val=id_pack_period.get(int(row[0]))
                    acs=False
                    for it in val:
                        if it[0]==row[5]:
                            acs=True
                            it.append(int(row[2]))
                    if acs==False:
                        temp=[]
                        temp.append(row[5])#название соц. сети
                        temp.append(int(row[2]))#время
                        id_pack_period[int(row[0])].append(temp)
                else:
                    id_pack_period[int(row[0])]=[]# каждый id принимает в значение свои транзакции
                    temp=[]
                    temp.append(row[5])#название соц. сети
                    temp.append(int(row[2]))#время
                    id_pack_period[int(row[0])].append(temp)
    now1 = datetime.now()
    print (now1-now)
    return id_pack_period