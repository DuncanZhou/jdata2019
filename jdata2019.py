import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import time
import math
import seaborn as sns
import lightgbm as lgb
import pickle
import os
import math
import gc

cache_path = '/data/ymzhou/jd/cache/'

path = "/data/ymzhou/jd/jdata/"

actions_path = "jdata_action.csv"
actions = pd.read_csv(path + actions_path,delimiter=",")

# 读取商品数据
product_path = "jdata_product.csv"
product = pd.read_csv(path + product_path,delimiter=",")

# 分割日期和时间
# actions['action_date'] = actions['action_time'].apply(lambda x : x.split(" ")[0])
# actions['action_time'] = actions['action_time'].apply(lambda x : x.split(" ")[1])

# 读取商家数据
shops_path = "jdata_shop.csv"
shop = pd.read_csv(path + shops_path,delimiter=",")

# 读取用户数据
users_path = "jdata_user.csv"
users = pd.read_csv(path + users_path,delimiter=",")

# 读取评论数据
comment_path = 'jdata_comment.csv'
comments = pd.read_csv(path + comment_path,delimiter=',')

# 统计一段时间内以<user_id,sku_id>对的统计特征
def _Get_Actions_Fea(actions,start_date,end_date,i):
    dump_path = cache_path + 'action_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path,'rb'))
    else:
        df = actions.copy()
        actions = actions[(actions['action_date'] >= start_date) & (actions['action_date'] <= end_date)][['user_id','sku_id','type','action_time']]
        temp = pd.get_dummies(actions['type'],prefix='%s-action' % str(i))
        # 将加入购物车操作列去除
        actions = pd.concat((actions,temp),axis=1)
        # 统计操作时段,活跃度特征
        temp = pd.get_dummies(actions['action_time'],prefix='%s-time' % str(i))
        actions = pd.concat((actions,temp),axis=1)
        actions = actions.groupby(['user_id','sku_id'],as_index=False).sum()
        
        actions = pd.merge(actions,product[['sku_id','shop_id','cate']],on='sku_id',how='left')
        
        # 统计用户这段时间内的购买力
        buy = actions[actions['type'] == 2].groupby('user_id',as_index=False)['sku_id'].count().rename(columns={'sku_id':'buy_times'})
        total = actions.groupby('user_id',as_index=False)['sku_id'].count().rename(columns={'sku_id':'op_times'})
        buy = pd.merge(total,buy,on='user_id',how='left')
        buy['%s-buy_ratio' % str(i)] = buy['buy_times'] / buy['op_times']
        del buy['buy_times']
        del buy['op_times']
        actions = pd.merge(actions,buy,on='user_id',how='left')
        # 统计商品被购买强度
        buy = actions[actions['type'] == 2].groupby('sku_id')['user_id'].nunique().reset_index().rename(columns={'user_id':'buy_times'})
        total = actions.groupby('sku_id')['user_id'].nunique().reset_index().rename(columns={'user_id':'op_times'})
        buy = pd.merge(total,buy,on='sku_id',how='left')
        buy['%s-product_buy_ratio' % str(i)] = buy['buy_times'] / buy['op_times']
        del buy['buy_times']
        del buy['op_times']
        actions = pd.merge(actions,buy,on='sku_id',how='left')
        # 商铺的购买率
        buy = actions[actions['type'] == 2].groupby('shop_id')['user_id'].nunique().reset_index().rename(columns={'user_id':'buy_times'})
        total = actions.groupby('shop_id')['user_id'].nunique().reset_index().rename(columns={'user_id':'op_times'})
        buy = pd.merge(total,buy,on='shop_id',how='left')
        buy['%s-shop_buy_ratio' % str(i)] = buy['buy_times'] / buy['op_times']
        del buy['buy_times']
        del buy['op_times']
        actions = pd.merge(actions,buy,on='shop_id',how='left')
        # 统计用户-商品购买率
        temp = _Get_BuyRatio_Fea(df,'sku_id',start_date,end_date)
        actions = pd.merge(actions,temp,on=['user_id','sku_id'],how='left')
        # 统计用户-cate购买率
        temp = _Get_BuyRatio_Fea(df,'cate',start_date,end_date)
        actions = pd.merge(actions,temp,on=['user_id','cate'],how='left')
        # 统计用户-shop购买率
        temp = _Get_BuyRatio_Fea(df,'shop_id',start_date,end_date)
        actions = pd.merge(actions,temp,on=['user_id','shop_id'],how='left')
        # 统计用户-cate-shop购买率
        temp = _Get_MultiBuyRatio_Fea(df,start_date,end_date)
        actions = pd.merge(actions,temp,on=['user_id','cate','shop_id'],how='left')

#         # 加入cate下的特征(直接在这里加入)
#         # 品类下的商品种类数
#         _fea6 = product.groupby('cate')['sku_id'].nunique().reset_index().rename(columns={'sku_id':'cate_pro_nums'})
#         actions = pd.merge(actions,_fea6,on='cate',how='left')
#         # 品类下的商铺数
#         _fea7 = product.groupby('cate')['shop_id'].nunique().reset_index().rename(columns={'shop_id':'cate_shop_nums'})
#         actions = pd.merge(actions,_fea7,on='cate',how='left')
        del actions['type']
        actions = actions.fillna(0)
        pickle.dump(actions, open(dump_path, 'wb'))
        gc.collect()
    return actions

def _Get_BuyRatio_Fea(actions,target_col,start_date,end_date):
    dump_path = cache_path + "%s-%s-%s-buy_ratio.pkl" % (start_date,end_date,target_col)
    if os.path.exists(dump_path):
        buy = pickle.load(open(dump_path,'rb'))
    else:
        df = actions[(actions['action_date'] >= start_date) & (actions['action_date'] <= end_date)]
        df = pd.merge(df,product[['sku_id','cate','shop_id']],on='sku_id',how='left')
        
        buy = df[df['type'] == 2].groupby(['user_id',target_col])['type'].count().reset_index().rename(columns={'type':'buy_times'})
        total = df.groupby(['user_id',target_col])['type'].count().reset_index().rename(columns={'type':'op_times'})
        buy = pd.merge(total,buy,on=['user_id',target_col],how='left')
        # 统计每个用户总操作次数
        temp = buy.groupby('user_id',as_index=False)['op_times'].sum().rename(columns={'op_times':'total_times'})
        buy = pd.merge(buy,temp,on='user_id',how='left')
        buy['%s-%s-%s-buy_ratio' % (start_date,end_date,target_col)] = buy['buy_times'] / buy['op_times']
        buy['%s-%s-%s-op_ratio' % (start_date,end_date,target_col)] = buy['op_times'] / buy['total_times']
        del buy['buy_times']
    #     del buy['op_times']
        del buy['total_times']
        buy = buy.rename(columns={'op_times':'%s-%s-%s-op_times' % (start_date,end_date,target_col)})
        pickle.dump(buy,open(dump_path,'wb'))
    return buy

# 统计用户-cate-shop购买力
def _Get_MultiBuyRatio_Fea(actions,start_date,end_date):
    dump_path = cache_path + "%s-%s-multi-buy_ratio.pkl" % (start_date,end_date)
    if os.path.exists(dump_path):
        buy = pickle.load(open(dump_path,'rb'))
    else:
        df = actions[(actions['action_date'] >= start_date) & (actions['action_date'] <= end_date)]
        df = pd.merge(df,product[['sku_id','cate','shop_id']],on='sku_id',how='left')
        
        buy = df[df['type'] == 2].groupby(['user_id','cate','shop_id'])['type'].count().reset_index().rename(columns={'type':'buy_times'})
        total = df.groupby(['user_id','cate','shop_id'])['type'].count().reset_index().rename(columns={'type':'op_times'})
        buy = pd.merge(total,buy,on=['user_id','cate','shop_id'],how='left')
        # 统计每个用户总操作次数
        temp = buy.groupby('user_id',as_index=False)['op_times'].sum().rename(columns={'op_times':'total_times'})
        buy = pd.merge(buy,temp,on='user_id',how='left')
        buy['%s-%s-multi-buy_ratio' % (start_date,end_date)] = buy['buy_times'] / buy['op_times']
        buy['%s-%s-multi-op_ratio' % (start_date,end_date)] = buy['op_times'] / buy['total_times']
        del buy['buy_times']
    #     del buy['op_times']
        del buy['total_times']
        buy = buy.rename(columns={'op_times':'%s-%s-multi-op_times' % (start_date,end_date)})
        pickle.dump(buy,open(dump_path,'wb'))
    return buy

def _Get_LongTerm_Fea(actions,train_end_date):
    start_days = '2018-02-01'
    # 用户的购买率
    dump_path = cache_path + '2018-02-01-%s-longterm-user.pkl' % train_end_date
    if os.path.exists(dump_path):
        _fea1 = pickle.load(open(dump_path,'rb'))
    else:
        df = actions[(actions['action_date'] >= start_days) & (actions['action_date'] <= train_end_date)]
        buy = df[actions['type'] == 2].groupby('user_id',as_index=False)['sku_id'].count().rename(columns={'sku_id':'buy_times'})
        total = df.groupby('user_id',as_index=False)['sku_id'].count().rename(columns={'sku_id':'op_times'})
        buy = pd.merge(total,buy,on='user_id',how='left')
        buy['%s-%s-user-buy_ratio' % (start_days,train_end_date)] = buy['buy_times'] / buy['op_times']
        del buy['buy_times']
        del buy['op_times']
        _fea1 = buy
        pickle.dump(_fea1,open(dump_path,'wb'))
    # 用户-sku_id购买率
    dump_path = cache_path + '2018-02-01-%s-longterm-user-sku.pkl' % train_end_date
    if os.path.exists(dump_path):
        _fea2 = pickle.load(open(dump_path,'rb'))
    else:
        _fea2 = _Get_BuyRatio_Fea(actions,'sku_id',start_days,train_end_date)
        pickle.dump(_fea2,open(dump_path,'wb'))
    # 用户-cate购买率
    dump_path = cache_path + '2018-02-01-%s-longterm-user-cate.pkl' % train_end_date
    if os.path.exists(dump_path):
        _fea3 = pickle.load(open(dump_path,'rb'))
    else:
        _fea3 = _Get_BuyRatio_Fea(actions,'cate',start_days,train_end_date)
        pickle.dump(_fea3,open(dump_path,'wb'))
    # 用户-shop购买率
    dump_path = cache_path + '2018-02-01-%s-longterm-user-shop.pkl' % train_end_date 
    if os.path.exists(dump_path):
        _fea4 = pickle.load(open(dump_path,'rb'))
    else:
        _fea4 = _Get_BuyRatio_Fea(actions,'shop_id',start_days,train_end_date)
        pickle.dump(_fea4,open(dump_path,'wb'))
    # 用户-cate-shop购买率
    dump_path = cache_path + '2018-02-01-%s-longterm-user-cate-shop.pkl' % train_end_date
    if os.path.exists(dump_path):
        _fea5 = pickle.load(open(dump_path,'rb'))
    else:
        _fea5 = _Get_MultiBuyRatio_Fea(actions,start_days,train_end_date)
        pickle.dump(_fea5,open(dump_path,'wb'))
    return _fea1,_fea2,_fea3,_fea4,_fea5

# 获取标签
def _Get_Label(actions,start_date,end_date):
    dump_path = cache_path + 'action_label_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path,'rb'))
    else:
        actions = actions[(actions['action_date'] >= start_date) & (actions['action_date'] <= end_date)]
        # 筛选出购买的user_id和sku_id
        actions = actions[actions['type'] == 2][['user_id','sku_id']]
        actions = actions.groupby(['user_id','sku_id'],as_index=False).sum()
        actions['label'] = 1
        actions = actions[['user_id','sku_id','label']]
        pickle.dump(actions,open(dump_path,'wb'))
    return actions

def _Get_Basic_Shop():
    dump_path = cache_path + 'basic_shop.pkl'
    if os.path.exists(dump_path):
        shop = pickle.load(open(dump_path,'rb'))
    else:
        shop = pd.read_csv(path + shops_path,delimiter=',')
        product = pd.read_csv(path + product_path,delimiter=",")
        # 去除cate为空的
    #     shop = shop[~shop['cate'].isnull()]
        # 统计每个商铺品类数
        temp = product.groupby('shop_id')['cate'].nunique().reset_index().rename(columns={'cate':'cate_nums'})
        shop = pd.merge(shop,temp,on='shop_id',how='left')
        # 统计每个商铺品牌数
        temp = product.groupby('shop_id')['brand'].nunique().reset_index().rename(columns={'brand':'brand_nums'})
        shop = pd.merge(shop,temp,on='shop_id',how='left')
        # 统计每个商铺商品数
        temp = product.groupby('shop_id')['sku_id'].nunique().reset_index().rename(columns={'sku_id':'product_nums'})
        shop = pd.merge(shop,temp,on='shop_id',how='left')
        shop = shop.fillna(-1)     
        pickle.dump(shop[['shop_id','fans_num','vip_num','shop_score','shop_reg_days',
                 'cate_nums','brand_nums','product_nums']],open(dump_path,'wb'))
    return shop

def _Get_Basic_User():
    dump_path = cache_path + 'basic_user.pkl' 
    if os.path.exists(dump_path):
        users = pickle.load(open(dump_path,'rb'))
    else:
        users = pd.read_csv(path + users_path,delimiter=",")
        # 填充缺失值
        users = users.fillna(-1)
        pickle.dump(users[['user_id','age','sex','user_lv_cd','city_level','province',
                 'city','county','user_reg_days']],open(dump_path,'wb'))
    return users

def _Get_Basic_Product():
    product = pd.read_csv(path + product_path,delimiter=",")
    return product[['sku_id','brand','market_time']]

def _Get_Basic_Comment(end_date):
    dump_path = cache_path + '%s-basic_comment.pkl' % end_date
    if os.path.exists(dump_path):
        comment = pickle.load(open(dump_path,'rb'))
    else:
        comments = pd.read_csv(path + comment_path)
        comment = comments[comments['dt'] <= end_date].groupby('sku_id',as_index=False)['good_comments','bad_comments'].sum()
        pickle.dump(comment[['sku_id','good_comments','bad_comments']],open(dump_path,'wb'))
    return comment

# 构造训练集

def _Generate_Train_Set(actions,train_start_date,train_end_date,label_start_date,label_end_date):
    print("窗口特征提取")
    
    dump_path = cache_path + 'train_set_%s_%s.pkl' % (train_start_date, train_end_date)
    if os.path.exists(dump_path):
        train = pickle.load(open(dump_path,'rb'))
    else:
        # 得到标签
        labels = _Get_Label(actions,label_start_date,label_end_date)
        
        # 获取训练时间内的特征
        train = None
        # 从训练截止时间往前推i天内的特征
        for i in (1,2,3,5,7,10,15,21,30):
            start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            if train is None:
                train = _Get_Actions_Fea(actions,start_days,train_end_date,i)          
            else:
                train = pd.merge(train,_Get_Actions_Fea(actions,start_days,train_end_date,i),
                                on=['user_id','sku_id','cate','shop_id'])
            
        # 合并长期特征
#             train = pd.merge(train,user_acc,on=[''])
        # 合并标签
        train = pd.merge(train,labels,on=['user_id','sku_id'],how='left')
        train = train.fillna(0)
        pickle.dump(train,open(dump_path,'wb'))
    
    # 加入其他特征(各维度购买/浏览转化率)    
    # 从训练截止时间往前推i天内的特征
    for i in (1,2,3,5,7,10,15,21,30):
        start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
        start_days = start_days.strftime('%Y-%m-%d')
        _user_fea,_sku_fea,_cate_fea,_shop_fea,_user_sku_fea,_user_cate_fea,_user_shop_fea,_user_cate_shop_fea\
        = _Get_Col_BuyRatio(actions,start_days,train_end_date)
        train = pd.merge(train,_user_fea,on=['user_id'],how='left')
        train = pd.merge(train,_sku_fea,on=['sku_id'],how='left')
        train = pd.merge(train,_cate_fea,on=['cate'],how='left')
        train = pd.merge(train,_shop_fea,on=['shop_id'],how='left')
        train = pd.merge(train,_user_sku_fea,on=['user_id','sku_id'],how='left')
        train = pd.merge(train,_user_cate_fea,on=['user_id','cate'],how='left')
        train = pd.merge(train,_user_shop_fea,on=['user_id','shop_id'],how='left')
        train = pd.merge(train,_user_cate_shop_fea,on=['user_id','cate','shop_id'],how='left')
        
#         # 添加收藏、评论、加购下单比
#         for _op_type in [3,4,5]:
#             _user_fea,_sku_fea,_cate_fea,_shop_fea,_user_sku_fea,_user_cate_fea,_user_shop_fea,_user_cate_shop_fea\
#             = _Get_Col_BuyRatio_ByType(actions,start_days,train_end_date,_op_type)
#             train = pd.merge(train,_user_fea,on=['user_id'],how='left')
#             train = pd.merge(train,_sku_fea,on=['sku_id'],how='left')
#             train = pd.merge(train,_cate_fea,on=['cate'],how='left')
#             train = pd.merge(train,_shop_fea,on=['shop_id'],how='left')
#             train = pd.merge(train,_user_sku_fea,on=['user_id','sku_id'],how='left')
#             train = pd.merge(train,_user_cate_fea,on=['user_id','cate'],how='left')
#             train = pd.merge(train,_user_shop_fea,on=['user_id','shop_id'],how='left')
#             train = pd.merge(train,_user_cate_shop_fea,on=['user_id','cate','shop_id'],how='left')
#             train = train.fillna(0)
    
    # 去除cate为空的
    train = train[~train['cate'].isnull()]
    
    # 长期特征
    fea1,fea2,fea3,fea4,fea5 = _Get_LongTerm_Fea(actions,train_end_date)
    # 合并长期特征
    train = pd.merge(train,fea1,on='user_id',how='left')
    train = pd.merge(train,fea2,on=['user_id','sku_id'],how='left')
    train = pd.merge(train,fea3,on=['user_id','cate'],how='left')
    train = pd.merge(train,fea4,on=['user_id','shop_id'],how='left')
    train = pd.merge(train,fea5,on=['user_id','cate','shop_id'],how='left')

    # 去除从4月8号到15号加入购物车用户
    print("静态特征提取")
    # 合并用户信息
    print("用户基本特征")
    _users = _Get_Basic_User()
    train = pd.merge(train,_users,on='user_id',how='left')
    
    # 合并产品信息 
    print("产品基本特征")
    _product = _Get_Basic_Product()
    train = pd.merge(train,_product,on='sku_id',how='left')
    
    # 合并商铺信息
    print("商铺基本特征")
    _shop = _Get_Basic_Shop()
    train = pd.merge(train,_shop,on='shop_id',how='left')
    
    # 合并评论
    print("评论基本特征")
    _comment = _Get_Basic_Comment(train_end_date)
    train = pd.merge(train,_comment,on='sku_id',how='left')

    users = train[['user_id','sku_id','shop_id','cate']].copy()
    labels = train['label'].copy()
    del train['user_id']
    del train['sku_id']
    del train['shop_id']
    del train['label']
    return users,train,labels

def _Generate_Test_Set(actions,train_start_date,train_end_date):
    print("窗口特征提取")
    
    dump_path = cache_path + 'train_set_%s_%s.pkl' % (train_start_date, train_end_date)
    if os.path.exists(dump_path):
        train = pickle.load(open(dump_path,'rb'))
    else:
        # 先统计长期特征
        start_days = '2018-02-01'
#         user_acc = _Get_Action_accumulate_Fea(start_days,train_end_date)
        # 得到标签
#         labels = _Get_Label(actions,label_start_date,label_end_date)
        
        # 获取训练时间内的特征
        train = None
        # 从训练截止时间往前推i天内的特征
        for i in (1,2,3,5,7,10,15,21,30):
            start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            if train is None:
                train = _Get_Actions_Fea(actions,start_days,train_end_date,i)
            else:
                train = pd.merge(train,_Get_Actions_Fea(actions,start_days,train_end_date,i),
                                on=['user_id','sku_id','cate','shop_id'])
        # 合并长期特征
#             train = pd.merge(train,user_acc,on=[''])
        # 合并标签
#         train = pd.merge(train,labels,on=['user_id','sku_id'],how='left')
        train = train.fillna(0)
        pickle.dump(train,open(dump_path,'wb'))
    
    # 加入其他特征(各维度购买/浏览转化率)    
    # 从训练截止时间往前推i天内的特征
    for i in (1,2,3,5,7,10,15,21,30):
        start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
        start_days = start_days.strftime('%Y-%m-%d')
        _user_fea,_sku_fea,_cate_fea,_shop_fea,_user_sku_fea,_user_cate_fea,_user_shop_fea,_user_cate_shop_fea\
        = _Get_Col_BuyRatio(actions,start_days,train_end_date)
        train = pd.merge(train,_user_fea,on=['user_id'],how='left')
        train = pd.merge(train,_sku_fea,on=['sku_id'],how='left')
        train = pd.merge(train,_cate_fea,on=['cate'],how='left')
        train = pd.merge(train,_shop_fea,on=['shop_id'],how='left')
        train = pd.merge(train,_user_sku_fea,on=['user_id','sku_id'],how='left')
        train = pd.merge(train,_user_cate_fea,on=['user_id','cate'],how='left')
        train = pd.merge(train,_user_shop_fea,on=['user_id','shop_id'],how='left')
        train = pd.merge(train,_user_cate_shop_fea,on=['user_id','cate','shop_id'],how='left')
        
        train = train.fillna(0)
        
#         # 添加收藏、评论、加购下单比
#         for _op_type in [3,4,5]:
#             _user_fea,_sku_fea,_cate_fea,_shop_fea,_user_sku_fea,_user_cate_fea,_user_shop_fea,_user_cate_shop_fea\
#             = _Get_Col_BuyRatio_ByType(actions,start_days,train_end_date,_op_type)
#             train = pd.merge(train,_user_fea,on=['user_id'],how='left')
#             train = pd.merge(train,_sku_fea,on=['sku_id'],how='left')
#             train = pd.merge(train,_cate_fea,on=['cate'],how='left')
#             train = pd.merge(train,_shop_fea,on=['shop_id'],how='left')
#             train = pd.merge(train,_user_sku_fea,on=['user_id','sku_id'],how='left')
#             train = pd.merge(train,_user_cate_fea,on=['user_id','cate'],how='left')
#             train = pd.merge(train,_user_shop_fea,on=['user_id','shop_id'],how='left')
#             train = pd.merge(train,_user_cate_shop_fea,on=['user_id','cate','shop_id'],how='left')
#             train = train.fillna(0)
            
    # 去除cate为空的
    train = train[~train['cate'].isnull()]
    
    # 合并长期特征
    # 长期特征
    fea1,fea2,fea3,fea4,fea5 = _Get_LongTerm_Fea(actions,train_end_date)
    train = pd.merge(train,fea1,on='user_id',how='left')
    train = pd.merge(train,fea2,on=['user_id','sku_id'],how='left')
    train = pd.merge(train,fea3,on=['user_id','cate'],how='left')
    train = pd.merge(train,fea4,on=['user_id','shop_id'],how='left')
    train = pd.merge(train,fea5,on=['user_id','cate','shop_id'],how='left')

    print("静态特征提取")
    # 合并用户信息
    print("用户基本特征")
    _users = _Get_Basic_User()
    train = pd.merge(train,_users,on='user_id',how='left')

    # 合并产品信息 
    print("产品基本特征")
    _product = _Get_Basic_Product()
    train = pd.merge(train,_product,on='sku_id',how='left')
    
    # 合并商铺信息
    print("商铺基本特征")
    _shop = _Get_Basic_Shop()
    train = pd.merge(train,_shop,on='shop_id',how='left')
    
    # 合并评论
    print("评论基本特征")
    _comment = _Get_Basic_Comment(train_end_date)
    train = pd.merge(train,_comment,on='sku_id',how='left')
    
    users = train[['user_id','sku_id','shop_id','cate']].copy()
    
    del train['user_id']
    del train['sku_id']
    del train['shop_id']
    return users,train

def _LGBSubmission():
    train_start_date = '2018-03-08'
    train_end_date = '2018-04-08'
    
    label_start_date = '2018-04-09'
    label_end_date = '2018-04-15'
    
    sub_start_date = '2018-03-15'
    sub_end_date = '2018-04-15'
    
    # 过滤掉仅有浏览记录且浏览次数小于3次的vender_id
#     df = _WithoutBuying(actions)
    user_index,train,labels = _Generate_Train_Set(actions,train_start_date,train_end_date,label_start_date,label_end_date)
#     user_index,train,labels = _Generate_Train_Set(df,train_start_date,train_end_date,label_start_date,label_end_date)
    print("Training")
#     print(list(train.columns))
    print(train.shape)
    train_data = lgb.Dataset(train.values,label=labels)
    num_round = 200
    lgb_param = {'num_leaves':31, 'num_trees':100, 'objective':'binary','metric':{'auc'},'learning_rate':0.05, \
             'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'boosting_type':'gbdt'}
    bst = lgb.train(lgb_param,train_data,num_round)
    
    print("feature_importance")
    importance = pd.DataFrame({
        'column': train.columns,
        'importance': bst.feature_importance(),
    }).sort_values(by='importance',ascending=False)[:200]   
    
    # 构造提交版本的数据
    sub_user_index,sub_train = _Generate_Test_Set(actions,sub_start_date,sub_end_date)
#     sub_user_index,sub_train = _Generate_Test_Set(df,sub_start_date,sub_end_date)
#     print(list(sub_train.columns))
    print("Predicting")
    pred = bst.predict(sub_train.values)
    sub_user_index['label'] = pred
    pred = sub_user_index[sub_user_index['label'] >= 0]
    # 一个用户有多个购买记录，选取预测概率大的
#     pred = pred.sort_values(by='label',ascending=False)
#     pred = pred.drop_duplicates(subset='user_id',keep='first')
    # 将类型转换为整型
    pred['user_id'] = pred['user_id'].astype(int)
    pred['cate'] = pred['cate'].astype(int)
    pred['shop_id'] = pred['shop_id'].astype(int)    
#     pred[['user_id','cate','shop_id']].to_csv('%.2fsubmission.csv' % threshold,index=False)
    return pred,importance

def submit(pred,threshold1,threshold2=None):
    # 0.035 12k  0.052
    # 0.033 14k 0.056
    pred = pred[pred['label'] >= threshold1]
    # 加入购物车用户预测结果
#     shopping_pred = _Get_Shopping_Users(actions)
#     pred = pd.concat((pred,shopping_pred[shopping_pred['label'] >= threshold2]))
    
    pred['user_id'] = pred['user_id'].astype(int)
    pred['cate'] = pred['cate'].astype(int)
    pred['shop_id'] = pred['shop_id'].astype(int) 
    pred = pred.drop_duplicates(subset=['user_id','cate','shop_id'],keep='first')
    print(pred.shape)
    pred[['user_id','cate','shop_id']].to_csv('submission.csv',index=False)
    return pred

# user_id、sku_id、cate、shop上的转化率(购买/浏览)
def _Get_Col_BuyRatio(actions,start_date,end_date):
    actions = actions[(actions['action_date'] >= start_date) & (actions['action_date'] <= end_date)]
    actions = pd.merge(actions,product[['sku_id','cate','shop_id']],on='sku_id',how='left')
    # user_id
    _user_fea = _Get_Buy_Watch_Ratio(actions,'user_id',start_date,end_date)    
    # sku_id
    _sku_fea = _Get_Buy_Watch_Ratio(actions,'sku_id',start_date,end_date)    
    # cate
    _cate_fea = _Get_Buy_Watch_Ratio(actions,'cate',start_date,end_date)    
    # shop
    _shop_fea = _Get_Buy_Watch_Ratio(actions,'shop_id',start_date,end_date)
    # user-sku
    _user_sku_fea = _Get_Multi_Buy_Watch_Ratio(actions,'sku_id',start_date,end_date)
    # user-cate
    _user_cate_fea = _Get_Multi_Buy_Watch_Ratio(actions,'cate',start_date,end_date)
    # user-shop
    _user_shop_fea = _Get_Multi_Buy_Watch_Ratio(actions,'shop_id',start_date,end_date)
    # user-cate-shop
    dump_path = cache_path + '%s-%s-user-cate-shop-buy-watch-ratio.pkl' % (start_date,end_date)
    if os.path.exists(dump_path):
        _user_cate_shop_fea = pickle.load(open(dump_path,'rb'))
    else:
        buy = actions[actions['type'] == 2].groupby(['user_id','cate','shop_id']).size().reset_index().rename(columns={0:'buy_times'})
        watch = actions[actions['type'] == 1].groupby(['user_id','cate','shop_id']).size().reset_index().rename(columns={0:'watch_times'})
        buy = pd.merge(buy,watch,on=['user_id','cate','shop_id'],how='left')
        buy['%s-%s-user-cate-shop-buy-watch-ratio' % (start_date,end_date)] = buy['buy_times'] / buy['watch_times']
        del buy['buy_times']
        del buy['watch_times']
        _user_cate_shop_fea = buy.fillna(0)
        pickle.dump(_user_cate_shop_fea,open(dump_path,'wb'))
    gc.collect()
    return _user_fea,_sku_fea,_cate_fea,_shop_fea,_user_sku_fea,_user_cate_fea,_user_shop_fea,_user_cate_shop_fea

def _Get_Buy_Watch_Ratio(actions,col,start_date,end_date):
    dump_path = cache_path + '%s-%s-%s-buy-watch-ratio.pkl' % (start_date,end_date,col)
    if os.path.exists(dump_path):
        _fea = pickle.load(open(dump_path,'rb'))
    else:
        buy = actions[actions['type'] == 2].groupby(col).size().reset_index().rename(columns={0:'buy_times'})
        watch = actions[actions['type'] == 1].groupby(col).size().reset_index().rename(columns={0:'watch_times'})
        buy = pd.merge(buy,watch,on=col,how='left')
        buy['%s-%s-%s-buy-watch-ratio' % (start_date,end_date,col)] = buy['buy_times'] / buy['watch_times']
        del buy['buy_times']
        del buy['watch_times']
        _fea = buy.fillna(0)
        pickle.dump(_fea,open(dump_path,'wb'))
        gc.collect()
    return _fea

def _Get_Multi_Buy_Watch_Ratio(actions,col,start_date,end_date):
    dump_path = cache_path + '%s-%s-%s-multi-buy-watch-ratio.pkl' % (start_date,end_date,col)
    if os.path.exists(dump_path):
        _fea = pickle.load(open(dump_path,'rb'))
    else:
        buy = actions[actions['type'] == 2].groupby(['user_id',col]).size().reset_index().rename(columns={0:'buy_times'})
        watch = actions[actions['type'] == 1].groupby(['user_id',col]).size().reset_index().rename(columns={0:'watch_times'})
        buy = pd.merge(buy,watch,on=['user_id',col],how='left')
        buy['%s-%s-%s-multi-buy-watch-ratio' % (start_date,end_date,col)] = buy['buy_times'] / buy['watch_times']
        del buy['buy_times']
        del buy['watch_times']
        _fea = buy.fillna(0)
        pickle.dump(_fea,open(dump_path,'wb'))
        gc.collect()
    return _fea

# user_id、sku_id、cate、shop上的转化率(购买/收藏，评论，加购)
def _Get_Col_BuyRatio_ByType(actions,start_date,end_date,op_type):
    actions = actions[(actions['action_date'] >= start_date) & (actions['action_date'] <= end_date)]
    actions = pd.merge(actions,product[['sku_id','cate','shop_id']],on='sku_id',how='left')
    # user_id
    _user_fea = _Get_Buy_Watch_Ratio_ByType(actions,'user_id',start_date,end_date,op_type)    
    # sku_id
    _sku_fea = _Get_Buy_Watch_Ratio_ByType(actions,'sku_id',start_date,end_date,op_type)    
    # cate
    _cate_fea = _Get_Buy_Watch_Ratio_ByType(actions,'cate',start_date,end_date,op_type)    
    # shop
    _shop_fea = _Get_Buy_Watch_Ratio_ByType(actions,'shop_id',start_date,end_date,op_type)
    # user-sku
    _user_sku_fea = _Get_Multi_Buy_Watch_Ratio_ByType(actions,'sku_id',start_date,end_date,op_type)
    # user-cate
    _user_cate_fea = _Get_Multi_Buy_Watch_Ratio_ByType(actions,'cate',start_date,end_date,op_type)
    # user-shop
    _user_shop_fea = _Get_Multi_Buy_Watch_Ratio_ByType(actions,'shop_id',start_date,end_date,op_type)
    # user-cate-shop
    dump_path = cache_path + '%s-%s-user-cate-shop-buy-%s-ratio.pkl' %\
    (start_date,end_date,str(op_type))
    if os.path.exists(dump_path):
        _user_cate_shop_fea = pickle.load(open(dump_path,'rb'))
    else:
        buy = actions[actions['type'] == 2].groupby(['user_id','cate','shop_id']).size().reset_index().rename(columns={0:'buy_times'})
        watch = actions[actions['type'] == op_type].groupby(['user_id','cate','shop_id']).size().reset_index().rename(columns={0:'watch_times'})
        buy = pd.merge(buy,watch,on=['user_id','cate','shop_id'],how='left')
        buy['%s-%s-user-cate-shop-buy-%s-ratio' % (start_date,end_date,str(op_type))] = buy['buy_times'] / buy['watch_times']
        del buy['buy_times']
        del buy['watch_times']
        _user_cate_shop_fea = buy.fillna(0)
        pickle.dump(_user_cate_shop_fea,open(dump_path,'wb'))
    gc.collect()
    return _user_fea,_sku_fea,_cate_fea,_shop_fea,_user_sku_fea,_user_cate_fea,_user_shop_fea,_user_cate_shop_fea

def _Get_Buy_Watch_Ratio_ByType(actions,col,start_date,end_date,op_type):
    dump_path = cache_path + '%s-%s-%s-buy-%s-ratio.pkl' % (start_date,end_date,col,str(op_type))
    if os.path.exists(dump_path):
        _fea = pickle.load(open(dump_path,'rb'))
    else:
        buy = actions[actions['type'] == 2].groupby(col).size().reset_index().rename(columns={0:'buy_times'})
        watch = actions[actions['type'] == op_type].groupby(col).size().reset_index().rename(columns={0:'watch_times'})
        buy = pd.merge(buy,watch,on=col,how='left')
        buy['%s-%s-%s-buy-%s-ratio' % (start_date,end_date,col,str(op_type))] = buy['buy_times'] / buy['watch_times']
        del buy['buy_times']
        del buy['watch_times']
        _fea = buy.fillna(0)
        pickle.dump(_fea,open(dump_path,'wb'))
        gc.collect()
    return _fea

def _Get_Multi_Buy_Watch_Ratio_ByType(actions,col,start_date,end_date,op_type):
    dump_path = cache_path + '%s-%s-%s-multi-buy-%s-ratio.pkl' %\
    (start_date,end_date,col,str(op_type))
    if os.path.exists(dump_path):
        _fea = pickle.load(open(dump_path,'rb'))
    else:
        buy = actions[actions['type'] == 2].groupby(['user_id',col]).size().reset_index().rename(columns={0:'buy_times'})
        watch = actions[actions['type'] == op_type].groupby(['user_id',col]).size().reset_index().rename(columns={0:'watch_times'})
        buy = pd.merge(buy,watch,on=['user_id',col],how='left')
        buy['%s-%s-%s-multi-buy-%s-ratio' % (start_date,end_date,col,str(op_type))] = buy['buy_times'] / buy['watch_times']
        del buy['buy_times']
        del buy['watch_times']
        _fea = buy.fillna(0)
        pickle.dump(_fea,open(dump_path,'wb'))
        gc.collect()
    return _fea

# 对某一维度构造序列特征（用户、商品、商铺、品类、用户-品类、用户-商铺、用户-品类-商铺），embedding
def _Get_Sequence_Fea(cols,actions,op_type,end_date,days=30,timesteps=1,dim=5):
    '''
    cols: 维度(列表形式)
    op_type: 操作类型
    days: 时间跨度
    timesteps : 步长，默认为1天
    dim： embedding后的维数
    '''
    from keras.models import Model
    from sklearn.preprocessing import MinMaxScaler
    from keras.layers import Dense,Input,RepeatVector,LSTM
    start_date = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=days)
    start_date = start_date.strftime('%Y-%m-%d')
    actions = actions[(actions['action_date'] >= start_date) & (actions['action_date'] <= end_date)]
    actions = pd.merge(actions,product[['sku_id','cate','shop_id']],on='sku_id',how='left')
    col_name = "_".join(cols)
    gb_cols = cols + ['action_date']
    sequence = actions[actions['type'] == op_type].groupby(gb_cols,as_index=False)['module_id'].count().rename(columns={'module_id':'%s_count' % col_name})
    sequence = pd.pivot_table(sequence,index=cols,columns='action_date',values='%s_count' % col_name).reset_index().fillna(0)
    data = sequence[sequence.columns.difference(cols)]
    # 归一化
    min_max_scaler = MinMaxScaler()
    data = min_max_scaler.fit_transform(data)

    # 转换成3维的
    data = np.asarray(data)

    data = data.reshape(data.shape[0],timesteps,-1)
    print(data.shape)
    # 设置输出维度
    latent_dim = dim
    input_dim = data.shape[-1]
    inputs = Input(batch_shape=(None,timesteps,input_dim))


    encoded = LSTM(latent_dim)(inputs)

    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(input_dim, return_sequences=True)(decoded)

    sequence_autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)

    print(encoder.summary())
    # 编译，训练

    sequence_autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    print("编译完成")
    sequence_autoencoder.fit(data, data,
                             epochs=10,
                             batch_size=80000,
                             shuffle=True)
    # 输出编码层结果
    encoder = encoder.predict(data)
    res = sequence[cols]
    for i in range(dim):
        res[col_name+"%d_%d_embedding" % (op_type,i)] = encoder[:,i]
    return res

def _Sequence_Embedding_Fea():
    cols = [['user_id'],['sku_id'],['cate'],['shop_id'],
           ['user_id','cate'],['user_id','shop_id'],['user_id','cate','shop_id']]
    for col in cols:
        dump_path = cache_path + "%s_embedding.pkl" % file_name
        if os.path.exists(dump_path):
            continue
        res = None
        for op_type in [1,2,3,4,5]:
            temp = _Get_Sequence_Fea(col,actions,op_type,'2018-04-15',days=30,dim=5)
            if res is None:
                res = temp
            else:
                res = pd.merge(res,temp,on=col,how='outer')
        res = res.fillna(0)
        # 持久化
        file_name = '_'.join(col)
        print(file_name)
        dump_path = cache_path + "%s_embedding.pkl" % file_name
        pickle.dump(res,open(dump_path,'wb')) 


def get_score(result,real):
    result1 = result[['user_id','cate']].drop_duplicates()
    tmp1 = pd.merge(real,result1,on=['user_id','cate'],how='inner')
    len1 = len(real)
    len2 = len(result1)
    len3 = len(tmp1)
    recall1 = len3*1.0/len1
    print('提交结果数：'+str(len2))
    print('正确结果数1：'+str(len3))
    print('recall1:'+str(recall1))
    precision1 = len3*1.0/len2
    print("precision1:"+str(precision1))
    score1 = 3*recall1*precision1/(2*recall1+precision1)
    
    result2 = result[['user_id','cate','shop_id']].drop_duplicates()
    tmp2 = pd.merge(real,result2,on=['user_id','cate','shop_id'],how='inner')
    len2 = len(result2)
    len3 = len(tmp2)
    recall2 = len3*1.0/len1
    precision2 = len3*1.0/len2
    score2 = 5*recall2*precision2/(2*recall2+3*precision2)
    score = 0.4*score1 + 0.6*score2
    print('提交结果数：'+str(len2))
    print('正确结果数2：'+str(len3))
    print('recall2:'+str(recall2))
    print("precision2:"+str(precision2))
    return score1,score2,score

def _Offline_Metric():
    from sklearn.model_selection import StratifiedKFold
    from sklearn import metrics
    train_start_date = '2018-03-03'
    train_end_date = '2018-04-03'
    
    label_start_date = '2018-04-04'
    label_end_date = '2018-04-11'
    num_round = 200
    lgb_param = {'num_leaves':31, 'num_trees':100, 'objective':'binary','metric':{'auc'},'learning_rate':0.05, \
             'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'boosting_type':'gbdt'}
    # 过滤掉仅有浏览记录且浏览次数小于3次的vender_id
#     df = _WithoutBuying(actions)
    train_path = cache_path + "mybaseline_train_set_%s_%s.pkl" % (train_start_date,train_end_date)
    if os.path.exists(train_path):
        user_index = pickle.load(open(cache_path + "mybaseline_train_set_user_%s_%s.pkl" % (train_start_date,train_end_date),'rb'))
        train = pickle.load(open(train_path,'rb'))
        labels = pickle.load(open(cache_path + "mybaseline_train_set_label_%s_%s.pkl" % (train_start_date,train_end_date),'rb'))
    else:
        user_index,train,labels = _Generate_Train_Set(actions,train_start_date,train_end_date,label_start_date,label_end_date)
        pickle.dump(user_index,open(cache_path + "mybaseline_train_set_user_%s_%s.pkl" % 
                                    (train_start_date,train_end_date),'wb'))
        pickle.dump(train,open(train_path,'wb'))           
        pickle.dump(labels,open(cache_path + "mybaseline_train_set_label_%s_%s.pkl" % 
                                (train_start_date,train_end_date),'wb')
                   )                   #     user_index,train,labels = _Generate_Train_Set(df,train_start_date,train_end_date,label_start_date,label_end_date)
    # 线下验证标准auc和score分数
    print("Training")
#     skf = StratifiedKFold(n_splits=5)
    
#     for train_id,test_id in skf.split(train,labels):
    train_data = lgb.Dataset(train.values,label=labels)

    bst = lgb.train(lgb_param,train_data,num_round)
    importance = pd.DataFrame({
        'column': train.columns,
        'importance': bst.feature_importance(importance_type='gain'),
    }).sort_values(by='importance',ascending=False)[:200]
    importance.to_csv("feature_importance200.csv",index=False)
    test_start_date = '2018-03-08'
    test_end_date = '2018-04-08'
    
    test_label_start_date = '2018-04-09'
    test_label_end_date = '2018-04-15'
    
    test_path = cache_path + "mybaseline_test_set_%s_%s.pkl" % (test_start_date,test_end_date)
    if os.path.exists(test_path):
        test_user_index = pickle.load(open(cache_path + "mybaseline_test_set_user_%s_%s.pkl" % (test_start_date,test_end_date),'rb'))
        test = pickle.load(open(test_path,'rb'))
        test_labels = pickle.load(open(cache_path + "mybaseline_test_set_label_%s_%s.pkl" % (test_start_date,test_end_date),'rb'))
    else:
        test_user_index,test,test_labels = _Generate_Train_Set(actions,test_start_date,test_end_date,test_label_start_date,test_label_end_date)
        pickle.dump(test_user_index,open(cache_path + "mybaseline_test_set_user_%s_%s.pkl" % 
                                    (test_start_date,test_end_date),'wb'))
        pickle.dump(test,open(test_path,'wb'))           
        pickle.dump(test_labels,open(cache_path + "mybaseline_test_set_label_%s_%s.pkl" % 
                                (test_start_date,test_end_date),'wb'))
    pred = bst.predict(test.values)
    # 先用auc评价
    print("auc is %.4f" % metrics.roc_auc_score(test_labels,pred))
    test = pd.concat((test_user_index[['user_id','shop_id']],test),axis=1)
    test['prob'] = pred
    test['label'] = test_labels
    pred = test[['user_id','cate','shop_id','prob']]
    pred = pred[pred['prob'] >= 0.033]
    # 将类型转换为整型
    pred['user_id'] = pred['user_id'].astype(int)
    pred['cate'] = pred['cate'].astype(int)
    pred['shop_id'] = pred['shop_id'].astype(int) 
    true_results = test[test['label'] == 1][['user_id','cate','shop_id']]
    # 线下得分
    f11,f12,score = get_score(pred[['user_id','cate','shop_id']],true_results)
    print("分数为%.4f,%.4f,%.4f" % (f11,f12,score))
