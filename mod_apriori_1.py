
# coding: utf-8

# In[23]:

from datetime import date
import csv

def all_packs():
    pcks=[]
    with open('trans.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='|', quotechar='|')
        pcks=[]
        spamreader.next()
        for row in spamreader:
            if (row[5] not in pcks):
                pcks.append(row[5])
        print "Всего паков: ",len(pcks)
        return pcks


# In[24]:

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


def runApriori(data_iter, minSupport, minConfidence, st_t, end_t, r_f_a, r_d, id_pack_period):
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
    #print len(id_pack_period)
    toRetRules = []
    id_p={}
    for key, v in id_pack_period.items():
        for t in v:
            tmpp=[]
            for itr in range(0,len(t)-1):
                if t[itr+1]>=st_t and t[itr+1]<=end_t:
                    tmpp.append(t[itr+1])
            if len(tmpp)!=0:
                if id_p.has_key(key)==False:
                    id_p[key]=[]
                tmpp_1=[]
                tmpp_1.append(t[0]) ### ключ : [[pack,[times]]    ]
                tmpp_1.append(sorted(tmpp))   ### сортирует по возрастанию
                id_p[key].append(tmpp_1)
    #print id_p
    
    temp_dict=[]
    
    
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
                        
                        count=0
                    
                        l_s=str(tuple(element))
                        l=l_s.strip("(").strip(")").strip(",").split(",")
                        for i in range(0,len(l)):
                            l[i]=l[i].strip("'")
                        
                        r_s=str(tuple(remain))
                        r=r_s.strip("(").strip(")").strip(",").split(",")
                        for i in range(0,len(r)):
                            r[i]=r[i].strip("'")

                        for k,v in id_p.items():# проверяем последовательность
                            accs=True
                            tmp=[]
                            for it in v:
                                tmp.append(it[0])# все паки у пользователя (в данном промежутке времени)
                            for j in l:
                                if j not in tmp:
                                    accs=False
                            for j in r:
                                if j not in tmp:
                                    accs=False
                            
                            if accs==True: #если у него все паки есть
                                min_left=None
                                max_right=None
                                for it in v:
                                    if it[0] in l:
                                        if min_left==None:
                                            min_left=it[1][0]
                                        else:
                                            if it[1][1]<min_left:
                                                min_left=it[1][0]
                                    if it[0] in r:
                                        if max_right==None:
                                            max_right=it[1][len(it[1])-1]
                                        else:
                                            if t[1][len(it[1])-1]>max_right:
                                                max_right=it[1][len(it[1])-1]
                                if max_right>=min_left:
                                    #temp_dict[k]=[]# создаем временный словарь айдишник - пак, если проходит по конфиденс то добавляем в текущий
                                    temp_dict.append(k)
                                        
                                    count += 1
                        confidence_lift[0]=(float(count)/(float(len(transactionList))*float(item_A)))#conf - 0
                        #print confidence_lift[0],",",
                        if confidence_lift[0] >= minConfidence:
                            #print temp_dict
                            inddd=str(len(r_d))
                            
                            date_for_period=str(date.fromtimestamp(st_t))+" to "+str(date.fromtimestamp(end_t))
                            
                            r_d[inddd]=[]
                            r_d[inddd].append(l_s+"==>"+r_s) #идентифицируем правила
                            r_d[inddd].append(date_for_period)#date
                            
                            for kkey in temp_dict: #добавляем появившиеся транзакции
                                if r_f_a.has_key(kkey)==True:
                                    r_f_a[kkey].append(inddd)
                                else:
                                    r_f_a[kkey]=[]
                                    r_f_a[kkey].append(inddd)
                            temp_dict=[]
                            
                            confidence_lift.append(getSupport(item)/(item_A*item_B))#lift - 1
                            if  item_A>item_B:
                                confidence_lift.append(item_B)#min(minsup) - 2
                            else:
                                confidence_lift.append(item_A)
                            #print "================="        #For checking
                            #print element,"---",item_A       #For checking
                            #print remain,"---",item_B      

                            #print item,"---",float(count)/float(len(transactionList)),"===",len(transactionList) #For checking
                            #print "\n"                       #For checking
                            #print str(tuple(element)),"===>",tuple(remain),"conf",confidence_lift[0]  #For checking
                            #print "=================\n\n"    #For checking
                            toRetRules.append(((tuple(element), tuple(remain)),
                                               confidence_lift))
                        else:
                            temp_dict=[]
    #print id_p
    return toRetItems, toRetRules, r_f_a, r_d



def dataFromFile(fname):
        """Function which reads from the file and yields a generator"""
        file_iter = open(fname, 'rU')
        for line in file_iter:
                line = line.strip().rstrip(',')                         # Remove trailing comma
                record = frozenset(line.split(','))
                yield record
                
def printResults(items, rules, rs, idc, t_d,start_date,end_date, len_ids, action_pack): 
                                        #принимает: итемы для минсапа, правила с конф, файл для записи результата,
                                        #название текущего пака, номер айдишника для записи в словарь данных по пакам (tableau),
                                        #словарь, дата начала и конца, кол-во транзакций в периоде
    """prints the generated itemsets sorted by support and the confidence rules sorted by confidence"""
    if len(rules)!=0:
        rs.write("SUP------------SUP------------SUP:\n")
        rs.write("For time boundaries on: "+"\n") #показывает периоды какого пака исследуются
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


    inFile = dataFromFile("fool.csv")

    minSupport = options.minS
    minConfidence = options.minC

    #items, rules = runApriori(inFile, minSupport, 0.2,0 , 10) #For checking

    #printResults(items, rules)


# In[25]:

def find_action_packs_with_periods(which, pcks): #which определяет какие именно паки - "all", "fb", "vk", "mm", "ok"
    action_pack={}
    allowed_packs=["fb","vk","mm","ok"]
    for it in pcks:
        if ( (which in allowed_packs) and (which in it[:2]) ) or which not in allowed_packs:
            times=[]
            # 1) составляем временные промежутки для паков

            with open('trans.csv', 'rb') as csvfile:
                spamreader = csv.reader(csvfile, delimiter='|', quotechar='|')
                spamreader.next()
                for row in spamreader:
                    if row[5]==it:
                        times.append(int(row[2]))

            difference=172800+43200 # = 2.5 days in seconds
            # сравниваем даты по timestamp
            # если покупки наблюдались в окрестности 2х дней, то
            # будем считать, что в эти дни акция действовала
            times.sort()

            start_time=[]
            end_time=[]

            beg=times[0]

            start_time.append(beg)
            for item in times:
                if beg+difference>=item:
                    beg=item
                else:
                    start_time.append(item)
                    end_time.append(beg)
                    beg=item
            end_time.append(beg)

            # 2) проверяем временные промежутки; если все промежутки лежат в рамках 2.5 недель,
            # то считаем пак акционным и проходимся априори алгоритмом по выявленным промежуткам

            week_time=604800 # week time
            access=True
            for i in range(0,len(start_time)):
                if end_time[i]-start_time[i] > week_time*2.5: # предполагаем, что акции длятся не более 2х с половиной недель
                    access=False
            if access==True:
                action_pack[it]=[]
                action_pack[it].append(start_time)
                action_pack[it].append(end_time)
    return action_pack
                
#action_pack


# In[26]:

from math import fabs
def periods_without_intersections(action_pack):
    one_day=86400 

    times_s=[]
    times_e=[]

    for it,time_st_end in action_pack.items():
        start_t=time_st_end[0]
        end_t=time_st_end[1]
        for i in range(0,len(time_st_end[0])):
            ind_to_delete=[] #индексы будут в  порядке возр
            acs=True

            for j in range(0,len(times_s)): #Убираем промежуток, если он входит в какой-либо другой или покрывает какой-либо другой
                if start_t[i]>=times_s[j] and end_t[i]<=times_e[j]:
                    acs=False
                elif start_t[i]<=times_s[j] and end_t[i]>=times_e[j]:
                    ind_to_delete.append(j)

            count=0      
            for ind in ind_to_delete:
                    times_s.pop(ind-count)
                    times_e.pop(ind-count)
                    count+=1
            ind_to_delete=[]

            if acs==True:
                acs1=False
                while acs1==False: # цикл потому что когда объеденили промежуток еще один промежуток мог стать близким |([ | ) ]. Когдаобъединили (] нам стал доступен |])
                    acs1=True
                    tmp1=None
                    tmp2=None
                    for j in range(0,len(times_s)): #Теперь у нас пересечения. Если где-нибудь начало или конец совпадают,
                                                    #то промежутки сливаются (он может совпадать с несколкими, сливаются первые попавшиеся)
                                                    #Если не совпадают, то оставляем промежуток.
                        #print start_t[i],times_s[j],len(times_s),j
                        if (fabs(start_t[i]-times_s[j]) <= one_day):
                            if tmp1==None:
                                ind_to_delete.append(j)
                                if start_t[i]-times_s[j]>=0:

                                    tmp1=times_s[j]
                                    tmp2=end_t[i]
                                else:
                                    tmp2=times_e[j]
                                    tmp1=start_t[i]
                        elif (fabs(end_t[i]-times_e[j]) <= one_day):
                            if tmp2==None:
                                ind_to_delete.append(j)
                                if end_t[i]-times_e[j]>=0:
                                    tmp1=times_s[j]
                                    tmp2=end_t[i]
                                else:
                                    tmp2=times_e[j]
                                    tmp1=start_t[i]

                    count=0      
                    for ind in ind_to_delete:
                        times_s.pop(ind-count)
                        times_e.pop(ind-count)
                        count+=1
                    ind_to_delete=[]
                    if tmp1!=None:
                        acs1=False
                        start_t[i]=tmp1
                        end_t[i]=tmp2

                times_s.append(start_t[i])
                times_e.append(end_t[i])
    return times_s, times_e


# In[27]:

def dict_id_pack_periods():
    id_pack_period={}
    with open('trans.csv', 'rb') as csvfile:
        spam = csv.reader(csvfile, delimiter='|', quotechar='|')
        spam.next()
        for row in spam:
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
        return id_pack_period


# In[29]:

from datetime import datetime
from datetime import date
def write_res_to_file_name(file_name, mnsp, mncf, t_d, r_f_a, r_d, id_pack_period, times_s, times_e, action_pack):
    id_count=0
    now = datetime.now()
    minSupport=mnsp
    minConfidence=mncf

    count=0
    with open(file_name, 'w') as rs:
        start_time=times_s
        end_time=times_e
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
                    csf.write(st+'\n')
            #now1 = datetime.now()
            #print "файлы: ","  ",(now1-now)

            inFile = dataFromFile("file_for_transactions.csv")
            
            items, rules, r_f_a, r_d = runApriori(inFile, minSupport, minConfidence, start_time[bound], end_time[bound], r_f_a, r_d, id_pack_period)  
            
            id_count, t_d=printResults(items, rules, rs, id_count, t_d, start_time[bound],end_time[bound],len(ids), action_pack)
            #print len(ids),"-",
            


    # 4) время работы алгоритма
    now1 = datetime.now()
    print (now1-now)
    return t_d, r_f_a, r_d


# In[ ]:


def runApriori_lvl3(data_iter, minSupport, minConfidence):
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


