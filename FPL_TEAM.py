#!/usr/bin/env python
# coding: utf-8

# In[5]:


import json
import pandas as pd
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns
import re
sns.set_style('whitegrid')
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 100)


# In[6]:


#source - https://fantasy.premierleague.com/drf/bootstrap-static
with open('data1.json', encoding = 'utf8') as data_file:    
    data = json.load(data_file)


# In[7]:


for i,j in data.items():
    pprint(i)


# In[8]:


data['stats_options']


# In[9]:


data['game-settings']


# In[10]:


player_data_json = data['elements']
print(player_data_json)


# In[11]:


player_data_df = pd.DataFrame(player_data_json)
pdata = player_data_df.copy()
pdata.head(10)


# In[12]:


drop_cols = ['chance_of_playing_this_round','chance_of_playing_next_round','code',
            'cost_change_event','cost_change_event_fall','cost_change_start',
            'cost_change_start_fall','dreamteam_count','ea_index','ep_this',
            'event_points','form','ict_index','in_dreamteam','loaned_in',
            'loaned_out','loans_in','loans_out','news','photo','special',
            'squad_number','status', 'transfers_out','transfers_in',
             'transfers_out_event','transfers_in_event','value_form','value_season']


# In[13]:


pdata.drop(drop_cols, axis = 1, inplace = True)
pdata.columns


# In[14]:


pdata['full_name'] = pdata.first_name + " " + pdata.second_name
pdata['elememt_type_name'] = pdata.element_type.map({x['id']:x['singular_name_short'] for x in data['element_types']})


# In[15]:


pdata.head()


# In[16]:


pdata.info()
pdata.shape


# In[17]:


'''pdata = pdata.loc(axis=0)[:,['full_name','first_name','second_name', 'element_type',
                     'element_type_name','id','team', 'team_code', 'web_name',
                     'saves','penalties_saved','clean_sheets','goals_conceded',
                     'bonus', 'bps','creativity','ep_next','influence', 'threat',
                     'goals_scored','assists','minutes', 'own_goals',
                     'yellow_cards', 'red_cards','penalties_missed',
                     'selected_by_percent', 'now_cost','points_per_game','total_points']]'''


# In[18]:


pdata['team'] = pdata.team.map({x['id']:x['name'] for x in data['teams']})


# In[19]:


pdata.head()


# In[20]:


pdata.corr()['total_points']


# In[21]:


pdata.pivot_table(index = 'elememt_type_name', values = 'total_points', aggfunc = np.mean)


# In[22]:


pdata.pivot_table(index='elememt_type_name', values='total_points', aggfunc=np.median)


# In[23]:


pdata.elememt_type_name.value_counts()


# In[24]:


f = plt.figure(figsize = (16,9))

ax1 = f.add_subplot(2,2,1)
ax2 = f.add_subplot(2,2,2, sharex = ax1, sharey = ax1)
ax3 = f.add_subplot(2,2,3, sharex = ax1, sharey = ax1)
ax4 = f.add_subplot(2,2,4, sharex = ax1, sharey = ax1)

ax1.set_title('GOALKEEPERS')
sns.distplot(pdata[pdata.elememt_type_name == 'GKP'].total_points, label = 'GKP', ax=ax1)
ax1.axvline(np.mean(pdata[pdata.elememt_type_name == 'GKP'].total_points), color = 'maroon', label = 'mean')

ax2.set_title('DEFENDERS')
sns.distplot(pdata[pdata.elememt_type_name == 'DEF'].total_points, label = 'DEF', ax = ax2)
ax2.axvline(np.mean(pdata[pdata.elememt_type_name == 'DEF'].total_points), color = 'maroon', label = 'mean')

ax3.set_title('MIDFIELDERS')
sns.distplot(pdata[pdata.elememt_type_name == 'MID'].total_points, label = 'MID', ax = ax3)
ax3.axvline(np.mean(pdata[pdata.elememt_type_name == 'MID'].total_points), color = 'maroon', label = 'mean')

ax4.set_title('FORWARDS')
sns.distplot(pdata[pdata.elememt_type_name == 'FWD'].total_points, label = 'FWD', ax = ax4)
ax4.axvline(np.mean(pdata[pdata.elememt_type_name == 'FWD'].total_points), color = 'maroon', label = 'mean')

plt.show()


# DISTRIBUTION OF PLAYERS BASED ON THEIR POSITIONS AND MEAN
# 

# In[25]:


len(pdata[(pdata.total_points==0)]) / pdata.shape[0]


# In[26]:


pdata[(pdata.total_points == 0)&(pdata.minutes == 0)]


# In[27]:


impute_cols = ['saves','penalties_saved', 'clean_sheets', 'goals_conceded', 'bonus', 'bps',
               'creativity', 'influence', 'threat', 'goals_scored','assists', 'minutes', 'own_goals',
               'yellow_cards', 'red_cards','penalties_missed','points_per_game', 'total_points']
positions = set(pdata.elememt_type_name)
costs = set(pdata.now_cost)
medians = {}
stds = {}


# In[28]:


for i in positions:
    medians['{}'.format(i)] = {}
    for c in costs:
        medians['{}'.format(i)]['{}'.format(c)] = {}
        for j in impute_cols:
            if pdata[(pdata.total_points!=0)&(pdata.minutes!=0)&(pdata.elememt_type_name==str(i))&(pdata.now_cost==c)].shape[0] > 0:
                median = np.median(pdata[(pdata.total_points!=0)&(pdata.minutes!=0)&(pdata.elememt_type_name==i)&(pdata.now_cost==c)][j].astype(np.float32))
                medians['{}'.format(i)]['{}'.format(c)]['{}'.format(j)] = median
            else:
                medians['{}'.format(i)]['{}'.format(c)]['{}'.format(j)] = 0


# In[29]:


for i in positions:
    stds['{}'.format(i)] = {}
    for c in costs:
        stds['{}'.format(i)]['{}'.format(c)] = {}
        for j in impute_cols:
            if pdata[(pdata.total_points!=0)&(pdata.minutes!=0)&(pdata.elememt_type_name==str(i))&(pdata.now_cost==c)].shape[0] > 0:
                std = np.std(pdata[(pdata.total_points!=0)&(pdata.minutes!=0)&(pdata.elememt_type_name==i)&(pdata.now_cost==c)][j].astype(np.float32))
                stds['{}'.format(i)]['{}'.format(c)]['{}'.format(j)] = std
            else:
                stds['{}'.format(i)]['{}'.format(c)]['{}'.format(j)] = 0


# In[30]:


for idx, row in pdata[(pdata.total_points==0)&(pdata.minutes==0)].iterrows():
    for col in impute_cols:
        pdata.loc[idx,col] = medians[str(row['elememt_type_name'])][str(row['now_cost'])][str(col)] + np.abs((np.random.randn()/1.5)*stds[str(row['elememt_type_name'])][str(row['now_cost'])][str(col)])


# In[31]:


pdata[pdata.full_name == 'Lewis Dunk']


# In[32]:


len(pdata[(pdata.total_points==0)]) / pdata.shape[0]


# In[33]:


f = plt.figure(figsize = (16,9))

ax1 = f.add_subplot(2,2,1)
ax2 = f.add_subplot(2,2,2, sharex = ax1, sharey = ax1)
ax3 = f.add_subplot(2,2,3, sharex = ax1, sharey = ax1)
ax4 = f.add_subplot(2,2,4, sharex = ax1, sharey = ax1)

ax1.set_title('GOALKEEPERS')
sns.distplot(pdata[pdata.elememt_type_name == 'GKP'].total_points, label = 'GKP', ax=ax1)
ax1.axvline(np.mean(pdata[pdata.elememt_type_name == 'GKP'].total_points), color = 'maroon', label = 'mean')

ax2.set_title('DEFENDERS')
sns.distplot(pdata[pdata.elememt_type_name == 'DEF'].total_points, label = 'DEF', ax = ax2)
ax2.axvline(np.mean(pdata[pdata.elememt_type_name == 'DEF'].total_points), color = 'maroon', label = 'mean')

ax3.set_title('MIDFIELDERS')
sns.distplot(pdata[pdata.elememt_type_name == 'MID'].total_points, label = 'MID', ax = ax3)
ax3.axvline(np.mean(pdata[pdata.elememt_type_name == 'MID'].total_points), color = 'maroon', label = 'mean')

ax4.set_title('FORWARDS')
sns.distplot(pdata[pdata.elememt_type_name == 'FWD'].total_points, label = 'FWD', ax = ax4)
ax4.axvline(np.mean(pdata[pdata.elememt_type_name == 'FWD'].total_points), color = 'maroon', label = 'mean')

plt.show()


# In[34]:


pdata.pivot_table(index='elememt_type_name', values='total_points', aggfunc=np.mean)


# In[35]:


pdata.pivot_table(index='elememt_type_name', values='total_points', aggfunc=np.median)


# In[43]:


#Linear Programming Library
from puLP import *


# In[ ]:





# In[ ]:




