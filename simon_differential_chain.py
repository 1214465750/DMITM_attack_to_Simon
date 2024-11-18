import numpy as np
from pysat.solvers import Cadical
from pysat.formula import IDPool, CNF
from time import time
from collections import Counter
from mpmath import mp, power
import math

def xor(s, list, const):
    length = len(list)
    newlist = np.array(list)
    # print('length',length)
    lis = np.zeros((2 ** length, length), dtype=np.bool_)
    for i in range(length):
        block_len = 2 ** (length - i - 1)
        block_num = 2 ** (i + 1)
        for j in range(block_num):
            if j % 2 == 1:
                lis[j * block_len:(j + 1) * block_len, i] = 1
    c = lis.sum(axis=1)
    d = c % 2

    e = np.array([-1 for i in range(length)])
    f = 1 - const
    for i in range(2 ** length):
        if d[i] == f:
            s.add_clause(((e ** lis[i]) * newlist).tolist())
    return None


def jiafaqi(m,vpool,list,value,list_cons):
    oneblock=len(list[0])
    block_num=len(list)

    length=oneblock*block_num
    newlist=[]

    for i in range(block_num):
        newlist+=list[i]
    # print(newlist)

    list_s=[None]*(length-1)
    for i in range(length-1):
        list_s[i]=[vpool.id() for i in range(min(i+1,value))]


    m.add_clause([-newlist[0],list_s[0][0]])
    m.add_clause([newlist[0], -list_s[0][0]])

    for i in range(1,value):
        for j in range(i):

            m.add_clause([newlist[i],list_s[i-1][j],-list_s[i][j]])
            m.add_clause([newlist[i], -list_s[i - 1][j], list_s[i][j]])

            m.add_clause([newlist[i],-list_s[i][i]])

            m.add_clause([-newlist[i],list_s[i][0]])

            m.add_clause([-newlist[i], list_s[i - 1][j], -list_s[i][j+1]])
            m.add_clause([-newlist[i], -list_s[i - 1][j], list_s[i][j+1]])

    for i in range(value,length-1):
        for j in range(value):

            m.add_clause([newlist[i], list_s[i - 1][j], -list_s[i][j]])
            m.add_clause([newlist[i], -list_s[i - 1][j], list_s[i][j]])
        for j in range(value-1):

            m.add_clause([-newlist[i], list_s[i][0]])

            m.add_clause([-newlist[i], list_s[i - 1][j], -list_s[i][j + 1]])
            m.add_clause([-newlist[i], -list_s[i - 1][j], list_s[i][j + 1]])

        m.add_clause([-list_s[i-1][value-1],-newlist[i]])


    m.add_clause([-newlist[length-1],-list_s[length-2][value-1]])


    newlist_cons=[]
    for i in range(len(list_cons)):
        newlist_cons.append(value-list_cons[-i-1])
    for i in range(len(list_cons)):

        if newlist_cons[i]+1<=len(list_s[(i+1) * oneblock - 1]):
            m.add_clause([-list_s[(i+1) * oneblock - 1][newlist_cons[i]]])

    m.add_clause([list_s[-1][148],newlist[-1]])
    m.add_clause([list_s[-1][147]])
    return newlist,list_s

def And(s, list, b):
    for i in list:
        s.add_clause([i, -b])
    lis = list.copy()
    for i in range(len(lis)):
        lis[i] = -list[i]
    s.add_clause(lis + [b])

    return None
def eq(s, list):
    for i in range(len(list) - 1):
        s.add_clause([list[i], -list[i + 1]])
        s.add_clause([-list[i], list[i + 1]])

def Or(s, list, b):
    s.add_clause(list + [-b])
    for i in list:
        s.add_clause([-i, b])
    return None

def rotate_left(lst, k):
    if not lst:
        return lst
    k = k % len(lst)  
    return lst[-k:] + lst[:-k]


def simon_diff_oneround(m,vpool,diff_l,diff_r,block_n,index,num_key,z,a,b,c,list_jia):
    var = [vpool.id() for i in range(block_n)]
    dou = [vpool.id() for i in range(block_n)]
    alpha = diff_l.copy()
    beta = [vpool.id() for i in range(block_n)]
    gama = [vpool.id() for i in range(block_n)]

    for i in range(block_n):
        Or(m, [alpha[(i - a) % block_n], alpha[(i - b) % block_n]], var[i])
        And(m, [alpha[(i - b) % block_n], -alpha[(i - a) % block_n], alpha[(i - (2 * a - b)) % block_n]], dou[i])
        xor(m, [gama[i], beta[i], alpha[(i - c) % block_n]], 0)

    m.add_clause([-i for i in alpha])

    for i in range(block_n):
        m.add_clause([-gama[i],var[i]])

    for i in range(block_n):
        m.add_clause([gama[i],-gama[(i-a+b)%block_n],-dou[i]])
        m.add_clause([-gama[i], gama[(i - a + b) % block_n], -dou[i]])

    outdiff_l=[vpool.id() for i in range(block_n)]
    for i in range(block_n):
        xor(m, [beta[i], diff_r[i], outdiff_l[i]], 0)

    weight=[vpool.id() for i in range(block_n)]
    for i in range(block_n):
        xor(m,[var[i],dou[i],weight[i]],0)
    list_jia.append(weight)

    return outdiff_l,diff_l

def jia_le_eq(s,vpool, list, k):

    if k > len(list):
        print('k的取值超过了list的个数，无效')
        return None
    if k < 0:
        print('错误 k小于0')
        return None
    if k==0:
        for i in list:
            s.add_clause([-i])
    else:
        length = len(list)
        z = [[vpool.id() for i in range(k)] for i in range(length - 1)]

        s.add_clause([-list[0], z[0][0]])
        for i in range(1, k):
            s.add_clause([-z[0][i]])

        for i in range(1, length - 1):
            s.add_clause([-list[i], z[i][0]])
            s.add_clause([-z[i - 1][0], z[i][0]])
            for j in range(1, k):
                s.add_clause([-list[i], -z[i - 1][j - 1], z[i][j]])
                s.add_clause([-z[i - 1][j], z[i][j]])
            s.add_clause([-list[i], -z[i - 1][k - 1]])
        s.add_clause([-list[length - 1], -z[length - 2][k - 1]])
        return None



def simon_diffchain(block_n,round,num_key,z,a,b,c,zuixiaozhi,list_cons):
    m=Cadical()
    vpool = IDPool()
    list_jia=[]
    list_diff_l=[None for i in range(round+1)]
    list_diff_r = [None for i in range(round + 1)]
    diff_l= [vpool.id() for i in range(block_n)]
    diff_r = [vpool.id() for i in range(block_n)]
    list_diff_l[0]=diff_l.copy()
    list_diff_r[0] = diff_r.copy()

    inl=0b0100010000000000000000000000000000000000000000000000000000000000
    inr=1
    outl=0b0001000000000000000000000000000000000000000000000000000000010000
    outr=0b0100010000000000000000000000000000000000000000000000000000000100



    for i in range(block_n):
        if (inl>>i)%2==1:
            m.add_clause([diff_l[i]])
        else:
            m.add_clause([-diff_l[i]])
        if (inr>>i)%2==1:
            m.add_clause([diff_r[i]])
        else:
            m.add_clause([-diff_r[i]])


    for  i in range(round):
        diff_l,diff_r=simon_diff_oneround(m,vpool,diff_l,diff_r,block_n,i,num_key,z,a,b,c,list_jia)
        list_diff_l[i+1]=diff_l.copy()
        list_diff_r[i + 1] = diff_r.copy()


    for i in range(block_n):
        if (outl >> i) % 2 == 1:
            m.add_clause([diff_l[i]])
        else:
            m.add_clause([-diff_l[i]])
        if (outr >> i) % 2 == 1:
            m.add_clause([diff_r[i]])
        else:
            m.add_clause([-diff_r[i]])

    jiafaqi(m,vpool,list_jia,zuixiaozhi,list_cons)


    sat = m.solve()
    list_weight=[]
    if sat==False:
        print('无解')
        exit()
    else:
        while(sat==True):
            jishu=0
            res = m.get_model()
            for i in list_jia:
                for j in i:
                    if res[j - 1] > 0:
                        jishu += 1
            list_weight.append(jishu)
            print(jishu)

            cons=[]

            for i in range(round+1):
                for j in range(block_n):
                    if res[list_diff_l[i][j]-1]>0:
                        cons.append(-list_diff_l[i][j])
                    else:
                        cons.append(list_diff_l[i][j])
                    if res[list_diff_r[i][j]-1]>0:
                        cons.append(-list_diff_r[i][j])
                    else:
                        cons.append(list_diff_r[i][j])
            m.add_clause(cons)
            sat=m.solve()

    count = Counter(list_weight)
    print(count)
    sum = 0
    for i in count:
        # print(i)
        # print(count[i])
        sum += 2 ** (-i) * count[i]
    print(math.log2(sum))
    return sat

def printvalue(res,index):
    if res[index-1]>0:
        print(end='1')
    else:
        print(end='0')




if __name__ == '__main__':
    list_version = [[16, 4, 32, 0], [24, 3, 36, 0], [24, 4, 36, 1], [32, 3, 42, 2], [32, 4, 44, 3], [48, 2, 52, 2],
                    [48, 3, 54, 3], [64, 2, 68, 2], [64, 3, 69, 3], [64, 4, 72, 4]]
    block_n, num_key, total_round, z_num = list_version[9]
    a=8
    b=1
    c=2
    z = [[1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0,
          1,
          0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, ],
         [1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1,
          1,
          0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, ],
         [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0,
          0,
          0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, ],
         [1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
          0,
          0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, ],
         [1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1,
          1,
          0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, ]]

    list_cons_48=[2,2,4,6,8,12,14,18,20,26,30,35,38,44,46,50,52,57,59,63,65,70,]
    list_cons_64 = [2, 2, 4, 6, 8, 12, 14, 18, 20, 26, 30, 36, 38, 44, 48, 54, 56, 62, 64, 66, 68, 72,74,78,80 ]
    list_cons_96 = [2, 2, 4, 6, 8, 12, 14, 18, 20, 26, 30, 36, 38, 44, 48, 54, 56, 62, 64, 66, 68, 72, 74, 78, 80, 86,
                    90, 96, 98, 104, 108, 114, 116, 122, 124, 126, 128]
    list_cons_128 = [2, 2, 4, 6, 8, 12, 14, 18, 20, 26, 30, 36, 38, 44, 48, 54, 56, 62, 64, 66, 68, 72, 74, 78, 80, 86,
                     90, 96, 98, 104, 108, 114, 116, 122, 124, 126, 128, 132, 134, 138, 140, 146, 150]

    zuixiaozhi=149
    round=41
    simon_diffchain(block_n, round, num_key, z[z_num], a, b, c, zuixiaozhi, list_cons_128[:round-1])


