import numpy as np
from collections import Counter



def gen_diff(posi,posj):
    
    diff=[]
    state=[]
    if posi==0:
        return diff,state
    else:
        if posj<block:
            diff,state=gen_diff(posi-1,posj+block)
            return diff,state
        else:
            diff.append([posi - 1, (posj + pa) % block + block])
            diff.append([posi - 1, (posj + pb) % block + block])
            diff.append([posi - 1, (posj + pc) % block + block])
            diff.append([posi - 1, posj  % block])
            state.append([posi - 1, (posj + pa) % block + block])
            state.append([posi - 1, (posj + pb) % block + block])
            return diff,state
def gen_state(posi,posj):
   
    state=[]
    key=[]
    if posi==0:
        return state,key
    else:
        if posj<block:
            state,key=gen_diff(posi-1,posj+block)
            return state,key
        else:
            state.append([posi - 1, (posj + pa) % block + block])
            state.append([posi - 1, (posj + pb) % block + block])
            state.append([posi - 1, (posj + pc) % block + block])
            state.append([posi - 1, posj  % block])
            key.append([posi - 1,posj  % block])

            return state,key
#通过差分得到涉及到的密钥
def  generate_keyfromdiff(posi,posj):
    

    diff=[[posi,posj]]
    state=[]
    key=[]
    while(diff!=[]):
        a=diff[0]
        diff=diff[1:]
        newdiff,newstate=gen_diff(a[0],a[1])
        diff+=newdiff
        state+=newstate
    while(state!=[]):
        a=state[0]
        state=state[1:]
        newstate, newkey=gen_state(a[0],a[1])
        state+=newstate
        key+=newkey
    return diff,state,key

def generate_keyfromstate(posi,posj):
    
    influent_key = []
    linear_key = []
    if posi == 0:
        return influent_key, linear_key
    else:
        if posj<block:
            influent_key, linear_key=generate_keyfromstate(posi-1,posj+block)
            return influent_key, linear_key
        else:
            I0, L0 = generate_keyfromstate(posi - 1, (posj - pa)%block+block)
            I1, L1 = generate_keyfromstate(posi - 1, (posj - pb)%block+block)
            I2, L2 = generate_keyfromstate(posi - 1, (posj - pc)%block+block)
            I3, L3 = generate_keyfromstate(posi - 1, posj%block)
            linear_key = decrease(L2 + L3 + [[posi - 1, posj % block]],[])
            influent_key = decrease(I0 + I1 + I2 + I3 + [[posi - 1, posj % block]],[])
            return influent_key, linear_key


def decrease( list_a, list_b):
    
    list = []
    for i in list_a:
        if i not in list_b and i not in list:
            list.append(i)
    return list

def int_to_np(a, block):
    b = np.zeros(block, dtype=bool)
    for i in range(block):
        b[i] = (a >> i) & 1
    return b
def print_diffmodel(flag, state, condition):
    if flag == 1:
        print('*', end='')
    else:
        
        if condition == 1:
            print('\033[31m%d\033[0m' % state, end='')
        else:
            print('%d' % state, end='')
def print_diffmodel012(a, b):
    if b==2 and a!=2:
        print('\033[31m%d\033[0m' % a, end='')
    else:
        print_xing(a)
def print_xing(state):
    if state==2:
        print('*',end='')
    else:
        print('%d' % state, end='')
def diff_model(diff_l,diff_r):

    diff_l = int_to_np(diff_l, block)
    diff_r = int_to_np(diff_r, block)
    
    state_l = np.zeros((round + 1, block), dtype=bool)
    state_r = np.zeros((round + 1, block), dtype=bool)
    flag_l = np.zeros((round + 1, block), dtype=bool)
    flag_r = np.zeros((round + 1, block), dtype=bool)
    
    state_l[round] = diff_l
    state_r[round] = diff_r
    
    for i in range(round - 1, -1, -1):
        state_l[i] = state_r[i + 1]
        flag_l[i] = flag_r[i + 1]
        flag_r[i] = (np.roll(flag_l[i], pa) | np.roll(flag_l[i], pb) | np.roll(state_l[i],
                                                                                                     pa) | np.roll(
            state_l[i], pb))
        state_r[i] = (~flag_r[i]) & (np.roll(state_l[i], pc) ^ state_l[i + 1])

    
    condition_bit = np.zeros((round + 1, block), dtype=bool)
    for i in range(round):
        condition_bit[i + 1] = flag_r[i] & (~flag_l[i + 1])

    
    for i in range(round + 1):
        
        for j in range(block-1,-1,-1):
            print_diffmodel(flag_l[i][j], state_l[i][j], condition_bit[i][j])
        print(end='  ')
        
        for j in range(block-1,-1,-1):
            print_diffmodel(flag_r[i][j], state_r[i][j], 0)
        print('')

    newstate_l = np.zeros((round + 1, block), dtype=int)
    newstate_r = np.zeros((round + 1, block), dtype=int)

    for i in range(round+1):
        for j in range(block):
            if flag_l[i][j]==True:
                newstate_l[i][j]=2
            else:
                if state_l[i][j]==True:
                    newstate_l[i][j] = 1
                else:
                    newstate_l[i][j] = 0

            if flag_r[i][j]==True:
                newstate_r[i][j]=2
            else:
                if state_r[i][j]==True:
                    newstate_r[i][j] = 1
                else:
                    newstate_r[i][j] = 0
    
    print('')
    for i in range(round + 1):
        # 左支
        for j in range(block - 1, -1, -1):
            print(newstate_l[i][j],end='')
        print(end='  ')
        # 右支 默认没有condition
        for j in range(block - 1, -1, -1):
            print(newstate_r[i][j], end='')
        print('')

    state_l=newstate_l
    state_r=newstate_r

    
    list_key=[]
    for i in range(round-1,-1,-1):
        for j in range(block):
            if state_r[i][j]==2:
                if state_l[i][(j-pa)%block]!=0:
                    inf,lin=generate_keyfromstate(i,(j - pb) % block+block)
                    list_key+=inf
                if state_l[i][(j - pb) % block] != 0:
                    inf, lin = generate_keyfromstate(i, (j - pa) % block + block)
                    list_key += inf
    list_key=decrease(list_key,[])
    print('需要猜测的主密钥比特',list_key)
    for i in range(round-1):
        for j in range(block-1,-1,-1):
            if [i,j] in list_key:
                print(end='1')
            else:
                print(end='0')
        print('')
    print('需要猜测的主密钥比特的个数',len(list_key))
    for (i, j) in list_key:
        print('$k_{%d}^{%d}$，' % (j, i), end='')
    print('')

    print('验证部分  知道的只有明文的取值和需要猜测密钥的取值 看能够恢复明文差分')
    message_l = np.zeros((round + 1, block), dtype=int)
    message_r = np.zeros((round + 1, block), dtype=int)
    key=np.zeros((4, block), dtype=int)


    
    message_l[0]=np.random.randint(0,2,block,dtype=int)
    message_r[0] = np.random.randint(0, 2, block, dtype=int)
    key=np.random.randint(0, 2, (4,block), dtype=int)

    for i in range(4):
        for j in range(block):
            key[i][j]=2

    for i in list_key:
        posi = i[0]
        posj = i[1]
        key[posi][posj]=0
    print('密钥取值 不知道取值的为2  知道的为0或1')
    for i in range(4):
        for j in range(block-1,-1,-1):
            print(key[i][j],end='')
        print('')
    print('状态取值 经过加密 不知道取值的为2  知道的为0或1')
    for i in range(round):
        message_r[i + 1] = message_l[i]
        for j in range(block):
            if  message_l[i][(j-pa)%block]!=2 and message_l[i][(j-pb)%block]!=2 and message_l[i][(j-pc)%block]!=2 and message_r[i][j]!=2 and key[i][j]!=2 :
                message_l[i+1][j]=(message_l[i][(j-pa)%block]&message_l[i][(j-pb)%block])^message_l[i][(j-pc)%block]^message_r[i][j]^key[i][j]
            else:
                message_l[i + 1][j]=2

    for i in range(round + 1):
        # 左支
        for j in range(block - 1, -1, -1):
            print(message_l[i][j], end='')
        print(end='  ')
        # 右支 默认没有condition
        for j in range(block - 1, -1, -1):
            print(message_r[i][j], end='')
        print('')

    ################################################################
    
    for i in range(round-1,-1, -1):
        for j in range(block):
            if state_r[i][j] ==2:
                if state_l[i][(j-pa)%block]==1 and state_l[i][(j-pb)%block]==1:
                    state_r[i][j] = message_l[i][(j - pa) % block] ^ message_l[i][(j - pb) % block] ^ 1 ^ state_l[i][(j - pc) % block] ^ state_l[i + 1][j]
                elif state_l[i][(j-pa)%block]==1 and state_l[i][(j-pb)%block]==0:
                    state_r[i][j] =  message_l[i][(j - pb) % block]  ^ state_l[i][(j - pc) % block] ^ state_l[i + 1][j]
                elif state_l[i][(j-pa)%block]==0 and state_l[i][(j-pb)%block]==1:
                    state_r[i][j] = message_l[i][(j - pa) % block] ^ state_l[i][(j - pc) % block] ^ state_l[i + 1][j]
                elif state_l[i][(j-pa)%block]==0 and state_l[i][(j-pb)%block]==0:
                    state_r[i][j] = state_l[i][(j - pc) % block] ^ state_l[i + 1][j]
                else:
                    print('程序有问题，')
                    print(i,j)
                    print(state_r[i][(j-pa)%block],state_r[i][(j-pb)%block])
                    exit()
        if i-1>=0:
            state_l[i-1]=state_r[i]

    
    print('各轮的差分取值 0或1 这是根据 明文和猜测部分密钥的取值决定的差分')
    for i in range(round + 1):
        
        for j in range(block - 1, -1, -1):
            print(state_l[i][j], end='')
        print(end='  ')
        
        for j in range(block - 1, -1, -1):
            print(state_r[i][j], end='')
        print('')
    ################################################################
    
    print('验证通过我的明文 部分猜测密钥的取值 得到的明文差分得到的明文对通过部分加密能否得到我的区分器差分')
    ml1 = message_l[0]
    mr1 = message_r[0]
    ml2 = ml1 ^ state_l[0]
    mr2 = mr1 ^ state_r[0]
    for i in range(round):
        ml1, mr1 = encone(ml1, mr1, key[i])
        ml2, mr2 = encone(ml2, mr2, key[i])
    print(ml1^ml2,mr1^mr2)
    ################################################################
    print('接下来是给概率固定部分中间差分的取值，来控制中间差分的传播，减少猜测的密钥数量')
    print('先输出 0 1 2 模式的差分')

    return None


def diff_model_dec(diff_l,diff_r):

    diff_l = int_to_np(diff_l, block)
    diff_r = int_to_np(diff_r, block)
    
    state_l = np.zeros((round + 1, block), dtype=bool)
    state_r = np.zeros((round + 1, block), dtype=bool)
    flag_l = np.zeros((round + 1, block), dtype=bool)
    flag_r = np.zeros((round + 1, block), dtype=bool)
    
    state_l[round] = diff_l
    state_r[round] = diff_r
    
    for i in range(round - 1, -1, -1):
        state_l[i] = state_r[i + 1]
        flag_l[i] = flag_r[i + 1]
        flag_r[i] = (np.roll(flag_l[i], pa) | np.roll(flag_l[i], pb) | np.roll(state_l[i],
                                                                               pa) | np.roll(
            state_l[i], pb))
        state_r[i] = (~flag_r[i]) & (np.roll(state_l[i], pc) ^ state_l[i + 1])

    
    condition_bit = np.zeros((round + 1, block), dtype=bool)
    for i in range(round):
        condition_bit[i + 1] = flag_r[i] & (~flag_l[i + 1])

    
    for i in range(round + 1):
        
        for j in range(block - 1, -1, -1):
            print_diffmodel(flag_l[i][j], state_l[i][j], condition_bit[i][j])
        print(end='  ')
        
        for j in range(block - 1, -1, -1):
            print_diffmodel(flag_r[i][j], state_r[i][j], 0)
        print('')

    newstate_l = np.zeros((round + 1, block), dtype=int)
    newstate_r = np.zeros((round + 1, block), dtype=int)

    for i in range(round + 1):
        for j in range(block):
            if flag_l[i][j] == True:
                newstate_l[i][j] = 2
            else:
                if state_l[i][j] == True:
                    newstate_l[i][j] = 1
                else:
                    newstate_l[i][j] = 0

            if flag_r[i][j] == True:
                newstate_r[i][j] = 2
            else:
                if state_r[i][j] == True:
                    newstate_r[i][j] = 1
                else:
                    newstate_r[i][j] = 0
    
    print('')
    for i in range(round + 1):
        
        for j in range(block - 1, -1, -1):
            print(newstate_l[i][j], end='')
        print(end='  ')
        
        for j in range(block - 1, -1, -1):
            print(newstate_r[i][j], end='')
        print('')

    state_l = newstate_l
    state_r = newstate_r

    
    list_key = []
    for i in range(round - 1, -1, -1):
        for j in range(block):
            if state_r[i][j] == 2:
                if state_l[i][(j - pa) % block] != 0:
                    inf, lin = generate_keyfromstate(i, (j - pb) % block + block)
                    list_key += inf
                if state_l[i][(j - pb) % block] != 0:
                    inf, lin = generate_keyfromstate(i, (j - pa) % block + block)
                    list_key += inf
    list_key = decrease(list_key, [])
    print('需要猜测的主密钥比特', list_key)
    print('需要猜测的主密钥比特的个数', len(list_key))

    qianlun=5
    zhonglun=22
    houlun=4
    all_round=qianlun+zhonglun+houlun-1
    list_key_dec=[]
    for (i,j) in list_key:
        list_key_dec.append([all_round-i,j])
    print('需要猜测的主密钥比特', list_key_dec)
    print('需要猜测的主密钥比特的个数', len(list_key_dec))
    for (i,j) in list_key_dec:
        print('$k_{%d}^{%d}$，'%(j,i),end='')
    print()

    return None


def fix_diff_model(diff_l,diff_r):
    print('接下来是给概率固定部分中间差分的取值，来控制中间差分的传播，减少猜测的密钥数量')
    print('先输出 0 1 2 模式的差分')


    diff_l = int_to_np(diff_l, block)
    diff_r = int_to_np(diff_r, block)
    
    state_l = np.zeros((round + 1, block), dtype=bool)
    state_r = np.zeros((round + 1, block), dtype=bool)
    flag_l = np.zeros((round + 1, block), dtype=bool)
    flag_r = np.zeros((round + 1, block), dtype=bool)
    
    state_l[round] = diff_l
    state_r[round] = diff_r
   
    for i in range(round - 1, -1, -1):
        state_l[i] = state_r[i + 1]
        flag_l[i] = flag_r[i + 1]
        flag_r[i] = (np.roll(flag_l[i], pa) | np.roll(flag_l[i], pb) | np.roll(state_l[i],
                                                                               pa) | np.roll(
            state_l[i], pb))
        state_r[i] = (~flag_r[i]) & (np.roll(state_l[i], pc) ^ state_l[i + 1])

    
    condition_bit = np.zeros((round + 1, block), dtype=bool)
    for i in range(round):
        condition_bit[i + 1] = flag_r[i] & (~flag_l[i + 1])

    
    for i in range(round + 1):
        
        for j in range(block - 1, -1, -1):
            print_diffmodel(flag_l[i][j], state_l[i][j], condition_bit[i][j])
        print(end='  ')
        
        for j in range(block - 1, -1, -1):
            print_diffmodel(flag_r[i][j], state_r[i][j], 0)
        print('')

    newstate_l = np.zeros((round + 1, block), dtype=int)
    newstate_r = np.zeros((round + 1, block), dtype=int)

    for i in range(round + 1):
        for j in range(block):
            if flag_l[i][j] == True:
                newstate_l[i][j] = 2
            else:
                if state_l[i][j] == True:
                    newstate_l[i][j] = 1
                else:
                    newstate_l[i][j] = 0

            if flag_r[i][j] == True:
                newstate_r[i][j] = 2
            else:
                if state_r[i][j] == True:
                    newstate_r[i][j] = 1
                else:
                    newstate_r[i][j] = 0
    
    print('')
    for i in range(round + 1):
        
        for j in range(block - 1, -1, -1):
            print(newstate_l[i][j], end='')
        print(end='  ')
        
        for j in range(block - 1, -1, -1):
            print(newstate_r[i][j], end='')
        print('')

    state_l = newstate_l
    state_r = newstate_r

    
    print('把中间差分*固定为0 控制差分')
    list_key=[]
    for i in range(round-1,-1,-1):
        for j in range(block):
            
            if state_r[i][j]==2:
                if state_l[i][(j-pa)%block]!=0:
                    inf,lin=generate_keyfromstate(i,(j - pb) % block+block)
                    list_key+=inf
                if state_l[i][(j - pb) % block] != 0:
                    inf, lin = generate_keyfromstate(i, (j - pa) % block + block)
                    list_key += inf
    list_key=decrease(list_key,[])
    print('需要猜测的主密钥比特',list_key)
    print('需要猜测的主密钥比特的个数',len(list_key))
    
    print('我现在就用14轮下部分的区分器 固定一个比特的*为0  然后延长一轮')
    
    for i in range(block):
        if state_r[round-1][i]==2:
            state_r[round - 1][i] =0
    for i in range(round-2,-1,-1):
        state_l[i] = state_r[i + 1]
        for j in range(block):

            if state_l[i][(j-pc)%block]==2 or state_l[i+1][j]==2 or state_l[i][(j-pa)%block]!=0 or state_l[i][(j-pb)%block]!=0:
                state_r[i][j]=2
            else:
                state_r[i][j] = state_l[i+1][j] ^ state_l[i][(j-pc)%block]





    print('固定*后的表')
    for i in range(round + 1):
        
        for j in range(block - 1, -1, -1):
            if i==0:
                print_xing(state_l[i][j])
            else:
                print_diffmodel012(state_l[i][j],state_r[i-1][j])
        print(end='  ')
        
        for j in range(block - 1, -1, -1):
            print_xing(state_r[i][j])
        print('')

    
    list_key = []
    for i in range(round - 1, -1, -1):
        for j in range(block):
            
            if state_r[i][j] == 2:
                if state_l[i][(j - pa) % block] != 0:
                    inf, lin = generate_keyfromstate(i, (j - pb) % block + block)
                    list_key += inf
                if state_l[i][(j - pb) % block] != 0:
                    inf, lin = generate_keyfromstate(i, (j - pa) % block + block)
                    list_key += inf
    list_key = decrease(list_key, [])
    print('需要猜测的主密钥比特', list_key)
    for (i,j) in list_key:
        print('$k_{%d}^{%d}$，'%(j,i),end='')
    print('\n需要猜测的主密钥比特的个数', len(list_key))

    
    qianlun = 4
    zhonglun = 22
    houlun = 3
    all_round = qianlun + zhonglun + houlun - 1
    list_key_dec = []
    for (i, j) in list_key:
        list_key_dec.append([all_round - i, j])
    print('需要猜测的主密钥比特', list_key_dec)
    print('需要猜测的主密钥比特的个数', len(list_key_dec))
    for (i, j) in list_key_dec:
        print('$k_{%d}^{%d}$，' % (j, i), end='')
    print()


def fix_diff_model_2(diff_l,diff_r):
    print('接下来是给概率固定部分中间差分的取值，来控制中间差分的传播，减少猜测的密钥数量')
    print('先输出 0 1 2 模式的差分')


    diff_l = int_to_np(diff_l, block)
    diff_r = int_to_np(diff_r, block)
    
    state_l = np.zeros((round + 1, block), dtype=bool)
    state_r = np.zeros((round + 1, block), dtype=bool)
    flag_l = np.zeros((round + 1, block), dtype=bool)
    flag_r = np.zeros((round + 1, block), dtype=bool)
    
    state_l[round] = diff_l
    state_r[round] = diff_r
    
    for i in range(round - 1, -1, -1):
        state_l[i] = state_r[i + 1]
        flag_l[i] = flag_r[i + 1]
        flag_r[i] = (np.roll(flag_l[i], pa) | np.roll(flag_l[i], pb) | np.roll(state_l[i],
                                                                               pa) | np.roll(
            state_l[i], pb))
        state_r[i] = (~flag_r[i]) & (np.roll(state_l[i], pc) ^ state_l[i + 1])

    
    condition_bit = np.zeros((round + 1, block), dtype=bool)
    for i in range(round):
        condition_bit[i + 1] = flag_r[i] & (~flag_l[i + 1])

    
    for i in range(round + 1):
        
        for j in range(block - 1, -1, -1):
            print_diffmodel(flag_l[i][j], state_l[i][j], condition_bit[i][j])
        print(end='  ')
        
        for j in range(block - 1, -1, -1):
            print_diffmodel(flag_r[i][j], state_r[i][j], 0)
        print('')

    newstate_l = np.zeros((round + 1, block), dtype=int)
    newstate_r = np.zeros((round + 1, block), dtype=int)

    for i in range(round + 1):
        for j in range(block):
            if flag_l[i][j] == True:
                newstate_l[i][j] = 2
            else:
                if state_l[i][j] == True:
                    newstate_l[i][j] = 1
                else:
                    newstate_l[i][j] = 0

            if flag_r[i][j] == True:
                newstate_r[i][j] = 2
            else:
                if state_r[i][j] == True:
                    newstate_r[i][j] = 1
                else:
                    newstate_r[i][j] = 0
    
    print('')
    for i in range(round + 1):
        
        for j in range(block - 1, -1, -1):
            print(newstate_l[i][j], end='')
        print(end='  ')
        
        for j in range(block - 1, -1, -1):
            print(newstate_r[i][j], end='')
        print('')

    state_l = newstate_l
    state_r = newstate_r

    ########################################################################
   
    print('把中间差分*固定为0 控制差分')
    list_key=[]
    for i in range(round-1,-1,-1):
        for j in range(block):
            
            if state_r[i][j]==2:
                if state_l[i][(j-pa)%block]!=0:
                    inf,lin=generate_keyfromstate(i,(j - pb) % block+block)
                    list_key+=inf
                if state_l[i][(j - pb) % block] != 0:
                    inf, lin = generate_keyfromstate(i, (j - pa) % block + block)
                    list_key += inf
    list_key=decrease(list_key,[])
    print('需要猜测的主密钥比特',list_key)
    print('需要猜测的主密钥比特的个数',len(list_key))
   
    print('我现在就用14轮下部分的区分器 固定一个比特的*为0  然后延长一轮')
    
    jishu=0
    for i in range(block):

        if state_r[round-1][i]==2:
            if jishu<2:
                state_r[round - 1][i] =0
                jishu+=1
    for i in range(round-2,-1,-1):
        state_l[i] = state_r[i + 1]
        for j in range(block):

            if state_l[i][(j-pc)%block]==2 or state_l[i+1][j]==2 or state_l[i][(j-pa)%block]!=0 or state_l[i][(j-pb)%block]!=0:
                state_r[i][j]=2
            else:
                state_r[i][j] = state_l[i+1][j] ^ state_l[i][(j-pc)%block]





    print('固定*后的表')
    for i in range(round + 1):
        
        for j in range(block - 1, -1, -1):
            if i==0:
                print_xing(state_l[i][j])
            else:
                print_diffmodel012(state_l[i][j],state_r[i-1][j])
        print(end='  ')
        
        for j in range(block - 1, -1, -1):
            print_xing(state_r[i][j])
        print('')

    ################################################################
    
    list_key = []
    for i in range(round - 1, -1, -1):
        for j in range(block):
            
            if state_r[i][j] == 2:
                if state_l[i][(j - pa) % block] != 0:
                    inf, lin = generate_keyfromstate(i, (j - pb) % block + block)
                    list_key += inf
                if state_l[i][(j - pb) % block] != 0:
                    inf, lin = generate_keyfromstate(i, (j - pa) % block + block)
                    list_key += inf
    list_key = decrease(list_key, [])
    print('需要猜测的主密钥比特', list_key)
    for (i,j) in list_key:
        print('$k_{%d}^{%d}$，'%(j,i),end='')
    print('\n需要猜测的主密钥比特的个数', len(list_key))

    
    qianlun = 5
    zhonglun = 22
    houlun = 4
    all_round = qianlun + zhonglun + houlun - 1
    list_key_dec = []
    for (i, j) in list_key:
        list_key_dec.append([all_round - i, j])
    print('需要猜测的主密钥比特', list_key_dec)
    print('需要猜测的主密钥比特的个数', len(list_key_dec))
    for (i, j) in list_key_dec:
        print('$k_{%d}^{%d}$，' % (j, i), end='')
    print()

def encone(l,r,key):
    return np.roll(l,pa)&np.roll(l,pb)^np.roll(l,pc)^key^r,l



if __name__ == '__main__':
    pa = 1
    pb = 8
    pc = 2
    block = 32
    round = 4

    # diff_model()
    # a,b,c=generate_keyfromdiff(4,0)
    # c=decrease(c,[])
    # print(len(c))
    # diff_model(0x40,0x11)
    # fix_diff_model_2(0x1000,0x4440)
    diff_model_dec(0x11,0x40)
    list1=[[0, 7], [0, 0], [0, 6], [1, 8], [0, 9], [0, 15], [1, 1], [0, 5], [1, 7], [2, 9], [0, 14], [0, 4], [1, 6], [0, 13], [1, 15], [0, 3], [1, 5], [2, 7], [0, 2], [0, 8], [1, 10], [0, 10], [1, 11], [0, 1], [1, 9], [0, 11], [0, 12]]
    #list2=[[0, 15], [0, 8], [0, 14], [1, 0], [0, 1], [0, 7], [1, 9], [0, 13], [1, 15], [2, 1], [0, 10], [0, 0], [1, 2], [0, 3], [0, 9], [1, 11], [1, 1], [2, 3], [0, 2], [0, 11], [1, 3], [0, 12], [1, 4], [0, 4], [1, 5], [0, 5], [0, 6]]
    list3=[[21, 15], [21, 8], [21, 14], [20, 0], [21, 1], [21, 7], [20, 9], [21, 13], [20, 15], [19, 1], [21, 10], [21, 0], [20, 2], [21, 3], [21, 9], [20, 11], [20, 1], [19, 3], [21, 2], [21, 11], [20, 3], [21, 12], [20, 4], [21, 4], [20, 5], [21, 5], [21, 6]]
