

'''


CODE USED TO IMPLEMENT A RECSYS USING IFHG



'''


from prefixspan import PrefixSpan
from sklearn.cluster import KMeans
import json
import os
import time
import pickle
import argparse
import pandas as pd
import tensorflow as tf
import scipy

#from tqdm import tqdm
from util import *
import json
import hypernetx as hnx
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from GraphLSHC import LSSHC
from spectral import spectral_clust
import math
#from fuzzy import fuzzify
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.preprocessing import RobustScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import skfuzzy as fuzz
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.spatial import distance
from sklearn.metrics import mean_squared_error
from sklearn.metrics import ndcg_score



def find_paths_containing_vertices(graph, vertices):
    paths = []
    valid_vertices = [vertex for vertex in vertices if vertex in graph.nodes]
    
    if len(valid_vertices) < 2:
        # Not enough valid vertices to find paths
        return paths
    for source in vertices:
        for target in vertices:
            if source != target:
                paths.extend(nx.all_simple_paths(graph, source=source, target=target))
    return paths
    
    
    
    

def ndcg_at_k(recommended, original, k):
    # Binary relevance scores (1 if the course is in the original list, 0 otherwise)
    relevance = [1 if course in original else 0 for course in recommended]
    
    if len(recommended)<k:
    	recommended.extend([0] * (k - len(recommended)))
    	
    if len(original)<k:
    	original.extend([0] * (k - len(original)))

    # Calculate DCG
    if len(relevance) < k:
        relevance.extend([0] * (k - len(relevance)))
        
    dcg = relevance[0] + np.sum(relevance[1:k] / np.log2(np.arange(2, k + 1)))

    # Calculate IDCG
    ideal_relevance = [1] * len(original)
    ideal_dcg = ideal_relevance[0] + np.sum(ideal_relevance[1:k] / np.log2(np.arange(2, k + 1)))

    # Calculate NDCG
    ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    return ndcg
    
    
    
    
def calculate_rel(course_id):
	#considers average rating, average course count, review positivity, instructor rating to calculate re;evance just like our previous work
	#for this the encoded course_id needs to be decoded
	
	#print("poly_reg_y_predicted",poly_reg_y_predicted)
	#id=data['course_id'].iloc[0].astype(int)
	#print("id",id)
	
	#print("course id before decoding is ",course_id)
	for key, value in item2idx.items():
    		if value == course_id:
        		real_id=key
        		break
        		
        
	car=data_reg.loc[data_reg['course_id'] ==real_id, 'n_course_avg_rating']
	cc=data_reg.loc[data_reg['course_id'] == real_id, 'n_Counts']
	#ip=data_reg.loc[data_reg['course_id'] == real_id, 'n_instructr_perf']
	
	if len(car)>0:
		car=car.iloc[0].item()
	elif len(car)==0:
		car=0
		#print("for courseid car is empty",real_id)
	if len(cc)>0:
		cc=cc.iloc[0].item()
	elif len(cc)==0:
		#print("for courseid cc is empty",real_id)
		cc=0
	#if len(ip)>0:
		#ip=ip.iloc[0].item()
	#elif len(ip)==0:
		#print("for courseid ip is empty",real_id)
		#ip=0
	
	lst1=[car]
	lst2=[cc]
	#lst3=[ip]
	
	
	df = pd.DataFrame(list(zip(lst1, lst2)),columns =['n_course_avg_rating', 'n_Counts'])
	#print("df",df.head())
	df=df.fillna(0)	
	df_fit=poly.fit_transform(df)
	predicted = poly_reg_model.predict(df_fit)
	rel=predicted.item()
	return car,cc,rel
	

def rmse(graph_data,rec_df):

	merged_data= graph_data.merge(rec_df, on=["learner_id","course_id"])
	merged_data['error']=merged_data['learner_rating']-merged_data['rel']
	#output_path='t_test.csv'
	#merged_data.to_csv(output_path, mode='a', header=not os.path.exists(output_path))

	return merged_data['error'].pow(2).mean()
	
	
def topological_sort(graph):
    def dfs(v, visited, stack):
        visited[v] = True
        for neighbor in graph.neighbors(v):
            if not visited[neighbor]:
                dfs(neighbor, visited, stack)
        stack.append(v)

    vertices = list(graph.nodes)
    visited = {v: False for v in vertices}
    stack = []

    for vertex in vertices:
        if not visited[vertex]:
            dfs(vertex, visited, stack)

    return stack[::-1]



def filter_rows_with_non_empty_columns(matrix, column_indices,unum):
    result_rows = {}

    for i,row in enumerate(matrix):
    	if i!=unum:
        	non_empty_count = sum(1 for col_idx in column_indices if row[col_idx] >0)
        	if non_empty_count > 0:
            		result_rows[i]=non_empty_count

    return result_rows	
    

def filter_rows_with_non_empty_columns2(matrix, column_indices,unum):
    result_rows = {}
    
    for i,row in enumerate(matrix):
    	if i!=unum:
        	course_listing=[col_idx for col_idx in column_indices if row[col_idx] >0]
        	non_empty_count = sum(1 for col_idx in column_indices if row[col_idx] >0)
        	if non_empty_count > 0:
            		result_rows[i]=course_listing

    return result_rows	
    
    	
	
	
	
def get_id2num(data, id):
    """
    Group by user and item, find the number of each group
    :param data:
    :param id:
    :return:
    """
    id2num = data[[id, 'learner_rating']].groupby(id, as_index=False)
    return id2num.size()
    
    
def encode_user_item(data, user2idx, item2idx):
    """
    Encode user and item
    :param data:
    :param user2idx:
    :param item2idx:
    :return:
    """
    data= data.astype({'learner_id': 'int32', 'course_id': 'int32', 'learner_rating': 'int32'})
    data['learner_id'] = data['learner_id'].map(user2idx).astype('int32')
    data['course_id'] = data['course_id'].map(item2idx).astype('int32')
    
    # Ensure learner_rating is int32
    data['learner_rating'] = data['learner_rating'].astype('int32')
    missing_users = data[~data['learner_id'].isin(user2idx.keys())]
    missing_courses = data[~data['course_id'].isin(item2idx.keys())]
    print("Missing Learner IDs:", missing_users)
    print("Missing Course IDs:", missing_courses)
    print("num of rows=",data.count()) 
    return data    

def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


#os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

print("*************************************************")
print("*************************************************")
#data=pd.read_csv("data/behaviour.csv")

data=pd.read_csv("ml1mdataextract.csv")
data= data.astype({'learner_id': 'int32', 'course_id': 'int32', 'learner_rating': 'int32'})
print("ML1M")
#data=data.drop(['learner_timestamp'],axis=1)#COMMENT FOR AMAZON DATASETS
#data=pd.read_csv("dataframe.csv")
data=data.drop(['Unnamed: 0'],axis=1)
#data.learner_comment = data.learner_comment.fillna('')


data = data.dropna()
data=data.sort_values(by=["learner_id"], ascending=True)
#data['learner_id'] = data['learner_id'].astype(int)
#TO FILTER OUT NOISY COURSES
course_freq = data['course_id'].value_counts()
frequency_threshold = 20
frequent_courses = course_freq[course_freq >= frequency_threshold].index.tolist()
data = data[data['course_id'].isin(frequent_courses)]



data=data.sort_values(by='learner_id', ignore_index=True)
data=data.iloc[0:50000,:]
data= data.astype({'learner_id': 'int32', 'course_id': 'int32', 'learner_rating': 'int32'})
print("STRUCTURE =",data.dtypes)
#print("DETAILED INFO=", data.info())
backup_data=data

#print("before",data.count())
#data=data.sort_values(by=["learner_timestamp"], ascending=True)
#data=data.drop(['learner_timestamp'],axis=1)
#data['course_id'] = data['course_id'].astype('int')

#print(data.loc[data['learner_id'] == 90453])
#data['learner_id'] = data['learner_id'].astype(int)
#print("i am printing it",data.isnull().values.any())

# Count the number of ratings for each user and item (the number of comments by each user, the number of comments received by each item)
user_id2num = get_id2num(data, 'learner_id')
item_id2num = get_id2num(data, 'course_id')

# Statistics user and item id
unique_user_ids = user_id2num.index
#print("unique_user_ids",unique_user_ids)
#unique_user_ids = user_id2num
    
unique_item_ids = item_id2num.index

#print("unique_user_ids",unique_user_ids)
#print("unique_item_ids ",unique_item_ids)
unique_user_ids = data['learner_id'].unique()
unique_item_ids = data['course_id'].unique()

# Create mappings for user IDs and item IDs
user2idx = {user_id: idx for idx, user_id in enumerate(unique_user_ids)}
item2idx = {item_id: idx for idx, item_id in enumerate(unique_item_ids)}

data['learner_id'] = data['learner_id'].map(user2idx)
data['course_id'] = data['course_id'].map(item2idx)

# Check for any missing values after mapping
missing_users = data['learner_id'].isnull().sum()
missing_courses = data['course_id'].isnull().sum()

#print("Missing Learner IDs after mapping:", missing_users)
#print("Missing Course IDs after mapping:", missing_courses)

# Output the modified DataFrame and its structure
#print("Modified DataFrame:\n", data.head())
#print("Structure after mapping:", data.dtypes)

data= data.astype({'learner_id': 'int32', 'course_id': 'int32', 'learner_rating': 'int32'})

print("num of rows=",data.count()) 


'''
user2idx={}
list1=data.learner_id.unique()
#print(list1)
for i in enumerate(unique_user_ids):
	user2idx[list1[i[0]]]=i[0];
    
    

item2idx={}
list2=data.course_id.unique()
#print(list1)
for i in enumerate(unique_item_ids):
	item2idx[list2[i[0]]]=i[0];

'''
# Encode raw data
#print("DATA INITIALLY")
#print(data)
data= data.astype({'learner_id': 'int32', 'course_id': 'int32', 'learner_rating': 'int32'})
#data = encode_user_item(data, user2idx, item2idx)

graph_data=data
#print("GRAPH DATA INITIALLY")
#print(graph_data)
#print(type(user2idx))

with open('convert.txt', 'w') as convert_file:
	convert_file.write(json.dumps(str(user2idx)))
	convert_file.write(json.dumps(str(item2idx)))

data.to_csv("dataframe.csv");
# Save user_num and item_num
usernum = len(unique_user_ids)
itemnum = len(unique_item_ids)
num = {"usernum": usernum, "itemnum": itemnum}

#print(usernum, itemnum)



# 1.Split training set




    
    
dataset = data_partition(data)
[user_train, user_valid, user_test, usernum, itemnum] = dataset
num_batch = len(user_train) // 10


rows = []

# Iterate through the dictionary
for learner_id, course_ids in user_train.items():
    # For each course_id in the list, create a tuple (learner_id, course_id) and append to the rows list
    rows.extend([(learner_id, course_id) for course_id in course_ids])

# Create a DataFrame from the list of rows
df_train = pd.DataFrame(rows, columns=['learner_id', 'course_id'])


rows = []

# Iterate through the dictionary
for learner_id, course_ids in user_test.items():
    # For each course_id in the list, create a tuple (learner_id, course_id) and append to the rows list
    rows.extend([(learner_id, course_id) for course_id in course_ids])

# Create a DataFrame from the list of rows
df_test= pd.DataFrame(rows, columns=['learner_id', 'course_id'])



cc = 0.0
max_len = 0
min_len=10000
newdict={}
for u in user_train:
    cc += len(user_train[u])
    newdict[u]=set(user_train[u])
    
    max_len = max(max_len, len(user_train[u]))
    min_len = min(min_len, len(user_train[u]))
    if len(user_train[u])==max_len:
    	max_u=u
    if len(user_train[u])==min_len:
    	min_u=u
    #if len(user_train[u])>=3:
    	#print(u," has", len(user_train[u]),"courses")
    	
    	
    	

#print("the user with min interactions",min_u)  	
#print("the user with maximum interactions is :",max_u)
    	


#print(newdict)


list_of_sets = list(newdict.values())
course_list_for_seq = [list(s) for s in list_of_sets] 
model = PrefixSpan(course_list_for_seq)

# Mine frequent sequential patterns
patterns = model.frequent(5)  # Adjust the support threshold as needed. Now it returns all patterns with support threshold >=3.


frequent_pattern_list=[]
# Print the discovered patterns
#print("PATTERNS")
#print(patterns)
for pattern in patterns:
    if len(pattern[1])>1:
    	frequent_pattern_list.append(pattern[1])
    
    
#print("frequent pattern list",frequent_pattern_list)    
   





unique_vertices = set(vertex for pattern in frequent_pattern_list for vertex in pattern)

# Step 2: Create a directed graph
G = nx.DiGraph()

# Step 3: Add edges based on sequential patterns
for pattern in frequent_pattern_list:
    G.add_edges_from(zip(pattern[:-1], pattern[1:]))





# Step 4: Apply transitive closure (optional)
G_transitive = nx.transitive_closure(G)
'''

largest_wcc = max(nx.weakly_connected_components(G), key=len)

# Create a subgraph containing only the largest weakly connected component
subgraph = G.subgraph(largest_wcc)

# Specify the layout for the subgraph (you can choose a different layout if needed)
pos = nx.spring_layout(subgraph)

# Draw the subgraph with the specified layout
nx.draw(subgraph, pos, with_labels=True, node_color='red', font_color='black')
plt.savefig("sequence_order_G.jpg")
plt.show()
'''



# Print the edges of the graph
#print("Edges of the graph:")
#for edge in G.edges:
    #print(edge)

# Print the edges of the transitive closure graph
#print("\nEdges of the transitive closure graph:")
#for edge in G_transitive.edges:
    #print(edge)
'''  
pos = nx.circular_layout(G_transitive)  # You can choose a different layout if needed
nx.draw(G_transitive,pos, with_labels=True, node_color='red', font_color='black')
plt.savefig("sequence_order_G_trans.jpg")
plt.show()
'''

    
    
partial_ordering=topological_sort(G_transitive)  
#print("PARTIAL ORDERING")
#print(partial_ordering)

#print("len(partial ordering)",   len(partial_ordering))
    
    
    
    
    
    
   
        
        
#dummydict={"S1":{"C2","C3","C4"},"S2":{"C4","C5","C6"},"S3":{"C2","C3","C6","C7"}}
#print("dummydict",dummydict)
#SSH = hnx.Hypergraph(dummydict, static=True)
#hnx.draw(SSH)
#plt.savefig("index.jpg")


graph_data["mem_degree"]=0
graph_data["non_degree"]=1

#finding the count of most frequent course
max_course_val=graph_data['course_id'].value_counts().nlargest(1)
max_course_count=max_course_val.values
#print("max_course_count",max_course_count[0])


#calculation membership changing parameter
mem=1/(max_course_count+1)



#updating membership and non membership degrees

#CONSIDER RATING TOO
for index,row in graph_data.iterrows():
	cid=row["course_id"]
	for index,row in graph_data.iterrows():
		if row["course_id"]==cid:
			graph_data.loc[index,"mem_degree"]=row["mem_degree"]+mem
			graph_data.loc[index,"non_degree"]=row["non_degree"]-mem
			
			


#creating weight value from mem degree and non degree

def weights(mem,rating,non):
	return (mem-non+(rating))/6


for index,row in graph_data.iterrows():
	graph_data.loc[index,"weight"]=weights(row["mem_degree"],row["learner_rating"],row["non_degree"])
	
	
#print("GRAPH_DATA")
#print(graph_data)


#create adjacency matrix1 of graph_data-edge vs nodes

cols = itemnum+1
rows = usernum+1

adj_mat=np.zeros(shape=(rows,cols))

for index,row in graph_data.iterrows():
	j=row["course_id"].astype(int)
	i=row["learner_id"].astype(int)
	w=row["weight"]
	adj_mat[i][j]=w
	
				


#print("ADJACENCY MATRIX")
#print(adj_mat)
#print(adj_mat.shape)

#print("\nThere are {0} users {1} items \n".format(usernum+1, itemnum+1))
#print("Average sequence length: {0}\n".format(cc / len(user_train)))
#print("Maximum length of sequence: {0}\n".format(max_len))
#print("Minimum length of sequence: {0}\n".format(min_len))

'''

#TRAINING
metrics=[]
#calculating course-weightages
#training polynomial regression model
data_reg=pd.read_csv("data_fuzzy.csv")#created using the code fuzzy_dataset.py
data_reg=data_reg[['learner_id','course_id','learner_rating','n_course_avg_rating','n_Counts','n_instructr_perf']]
#data_reg=data_reg[['n_course_avg_rating','n_Counts','n_instructr_perf','learner_rating']]
data_reg=data_reg.dropna()
X, y = data_reg[['n_course_avg_rating','n_Counts','n_instructr_perf']], data_reg["learner_rating"]
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.05, random_state=42)
poly_reg_model = LinearRegression()
poly_reg_model.fit(X_train, y_train)
#print("X",X_test)
poly_reg_y_predicted = poly_reg_model.predict(X_test)

for no_of_clust in range(5,usernum-3,1):


	#print("no of clusters=",no_of_clust)

	#start with clustering 

	lsshc=LSSHC(mode_framework='eigen_trick', mode_Z_construction='knn', mode_Z_construction_knn='random',
                 mode_sampling_schema='HSV', mode_sampling_approach='random', m_hyperedge=usernum+1, l_hyperedge =math.ceil((usernum+1)*.5),
                 knn_s=5, k=no_of_clust)
	label,n_features,centers=lsshc.fit(adj_mat)

	train_list=train_data.learner_id.unique().tolist()
	tp=0
	fp=0
	total=0

	for unum in train_list:
		#print("current user:",unum)
		total=total+1
		
		label_minu=label[unum]
		#print("the cluster label assigned to {} is {}".format(min_u,label_minu))



		#ranking other clusters based on Euclidian distance


		distances=[]
		for i in range(0,no_of_clust):
			distances.append(math.dist(centers[label_minu],centers[i]))


		sorted_distances=np.argsort(distances)
		#print("clusters after ranking",sorted_distances)	

		#courses done by learners in same cluster

		courses_for_minu=[]
		for i in range (0,usernum+1):
			if label[i]==label[unum]:
				course_list=graph_data.loc[graph_data['learner_id'] == i, 'course_id'].tolist()
				courses_for_minu.extend(course_list)

		#minu_courses=graph_data.loc[graph_data['learner_id'] == unum, 'course_id'].tolist()#courses already done by min_u



		#removing courses already done by min_u from the rec list

		#courses_for_minu= [x for x in courses_for_minu if x not in minu_courses]
		#courses_for_minu= [*set(courses_for_minu)]
		#print("course recs for {}:".format(min_u))
		#print(courses_for_minu)


		
		rec_list = []
		for course in courses_for_minu:
			course_id = course  
			carval,countval,ipval,rel= calculate_rel(course_id)
	
			rec_list.append([course_id,carval,countval,ipval,rel])

		rec_df = pd.DataFrame(rec_list, columns=['course_id','CAR','CC','IP','rel'])
		rec_df = rec_df.sort_values(by=['rel'], ascending=False)
		#print(rec_df)
		#making a list out of top-k courses

		rec_list = rec_df.course_id.values.tolist()
		rec_list=rec_list[0:no_of_clust]



		course_list_original=train_data.loc[train_data['learner_id'] == unum, 'course_id'].tolist()
		#print(course_list_original)


		#cheking if the courses recommended are actually done by the learner
		set1=set(rec_list)
		#print("set1",set1)
	
		set2=set(course_list_original)
		
		#print("set2",set2)
		
		fn=len(set2.difference(set1))
		if (set1 & set2):
      			tp=tp+1
      		
		else:
      			fp=fp+1
      			
      			
      			
      			
      		#APPLYING THE FIGURED OUT PARTIAL ORDER RELATIONSHIPS
		partial_ordered_rec_list = []
		nval=topn
		for vertex in partial_ordering:
			if vertex in rec_list:
				partial_ordered_rec_list.append(vertex)
				nval -= 1
			if nval == 0:
				break
			
		#in case the list is too small:
	
		partial_ordered_rec_list.extend(rec_list)
		partial_ordered_rec_list=set(partial_ordered_rec_list)
		partial_ordered_rec_list=list(partial_ordered_rec_list)
		partial_ordered_rec_list=partial_ordered_rec_list[0:topn]
		
		NDCG_partial+=ndcg_val
		if best_ndcg_partial<ndcg_val and ndcg_val<1:
			best_ndcg_partial=ndcg_val
      		
      		
      		set_partial=set(partial_ordered_rec_list)
      		try:
			p_partial=p_partial+len(set_partial & set2)/len(set_partial)
			r_partial=r_partial+len(set_partial & set2)/len(course_list_original)
			
		except:
			print("division by zero")
			
			
			
	precision =tp/(tp+fp)
	recall=tp/(tp+fn)
	metrics.append([no_of_clust,tp,fp,fn,precision,recall])

	metrics_df = pd.DataFrame(metrics, columns=['CLUSTERS','TP','FP','FN','PRECISION','RECALL'])


metrics_df = metrics_df.sort_values(by=['PRECISION','RECALL'], ascending=False)





no_of_clust=metrics_df['CLUSTERS'].iloc[0]
print("no_of_clust",no_of_clust)
num = {"no_of_clust": int(no_of_clust)}
json.dump(num, open(os.path.join('user_item_num.json'), 'w'))
#print('save: user_item_num.json')




#END OF TRAINING





'''


#no_of_clust=data['no_of_clust']
no_of_clust=usernum-5





topn=20
#no_of_clust=8
#print("optimal number of clusters =:",no_of_clust)
lsshc=LSSHC(mode_framework='eigen_trick', mode_Z_construction='knn', mode_Z_construction_knn='random',
                 mode_sampling_schema='HSV', mode_sampling_approach='random', m_hyperedge=usernum+1, l_hyperedge =math.ceil((usernum+1)*.5),
                 knn_s=5, k=no_of_clust)
X=lsshc.fit(adj_mat)

kmeans = KMeans(n_clusters=no_of_clust, random_state=0).fit(X)
centers=kmeans.cluster_centers_
#print("CLUSTER CENTERS:",kmeans.cluster_centers_)
#inertia = kmeans.inertia_
label = kmeans.labels_
#print("INERTIA:",inertia)
n_features=kmeans.n_features_in_
#print("FEATURES:",n_features)






#TESTING
#TOP-N PREDICTIONS
#Top n predictions for the users in test data 
data_reg=pd.read_csv("ml1mdata_fuzzy.csv")#created using the code fuzzy_dataset.py
#data_reg=pd.read_csv("behaviourdata_fuzzy.csv")#created using the code fuzzy_dataset.py
data_reg=data_reg[['learner_id','course_id','learner_rating','n_course_avg_rating','n_Counts']]
#data_reg=data_reg[['n_course_avg_rating','n_Counts','n_instructr_perf','learner_rating']]
data_reg=data_reg.dropna()
X, y = data_reg[['n_course_avg_rating','n_Counts']], data_reg["learner_rating"]
poly = PolynomialFeatures(degree=5, include_bias=False)
poly_features = poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.05, random_state=42)
poly_reg_model = LinearRegression()
poly_reg_model.fit(X_train, y_train)
#print("X",X_test)
poly_reg_y_predicted = poly_reg_model.predict(X_test)

test_list= list(user_test.keys())
train_list=list(user_train.keys())
'''
test_list=test_data.learner_id.unique().tolist()
train_list=test_data.learner_id.unique().tolist()
'''
tp=0
fp=0
total=0
ndcg_tot=0
NDCG=0
HR=0
HR_partial=0
rmseval=0
p=0
r=0
best_ndcg=0
NDCG_partial=0
agg_p=0
agg_p_partial=0
agg_r=0
agg_r_partial=0
p_partial=0
r_partial=0
best_ndcg_partial=0
best_p=0
best_r=0
best_p_partial=0
best_r_partial=0
for unum in test_list:
	
	course_list_original=df_test.loc[df_test['learner_id'] == unum, 'course_id'].tolist()
	if len(course_list_original)<topn:
		continue
	course_list_ultimate=graph_data.loc[graph_data['learner_id']==unum,'course_id'].tolist()
	flag=1
	#print("current user:",unum)
	total=total+1
	rank=0
	
	
	label_minu=label[unum]
	#print("the cluster label assigned to {} is {}".format(min_u,label_minu))



	#ranking other clusters based on Euclidian distance
	'''
	
	distances=[]
	for i in range(0,no_of_clust):
		distances.append(math.dist(centers[label_minu],centers[i]))

	
	'''
	distances=[]
	for i in range(0,no_of_clust):
		distances.append(scipy.spatial.distance.minkowski(centers[label_minu],centers[i], p=2, w=None))
		
	
	
	
	sorted_distances=np.argsort(distances)
	
	#print("clusters after ranking",sorted_distances)	

	#courses done by learners in same cluster

	courses_for_minu=[]
	for i in sorted_distances:
		for k in train_list:
			if i==label[k]:
				course_list=df_train.loc[df_train['learner_id'] == k, 'course_id'].tolist()
				courses_for_minu.extend(course_list)
				if len(courses_for_minu)>=topn:
					flag=0
					break
		if flag==0:
			break
			
					

	minu_courses=df_train.loc[df_train['learner_id'] == unum, 'course_id'].tolist()#courses already done by min_u

	if len(courses_for_minu)<topn:
		print("INSUFFICIENT NUMBER OF COURSES.. TAKE NEAREST CLUSTERS TOO")

	#removing courses already done by min_u from the rec list

	#courses_for_minu= [x for x in courses_for_minu if x not in minu_courses]
	#courses_for_minu= [*set(courses_for_minu)]
	#print("course recs for {}:".format(min_u))
	#print(courses_for_minu)


	#calculating course-weightages

	rec_list = []
	for course in courses_for_minu:
		course_id = course  
		carval,countval,rel= calculate_rel(course_id)
	
		rec_list.append([unum,course_id,carval,countval,rel])

	rec_df = pd.DataFrame(rec_list, columns=['learner_id','course_id','CAR','CC','rel'])
	#rec_df = rec_df.sort_values(by=['rel'], ascending=False)
	#print(rec_df)
	
	#comparing ratings to generate RMSE values
	rm=rmse(graph_data,rec_df)
	
	if math.isnan (rm):
		rm=0
	rmseval=rmseval+rm
	#print("rmseval",rmseval)
	#making a list out of top-k courses

	rec_list = rec_df.course_id.values.tolist()
	part_rec_list=rec_list
	rec_list=rec_list[0:topn]


	vertices=df_train.loc[df_train['learner_id'] == unum, 'course_id'].tolist()
	#paths_containing_vertices=find_paths_containing_vertices(G_transitive , vertices)
	
	#print(course_list_original)
	'''
	for cour in course_list_original:
		if cour in rec_list:
			rank=rec_list.index(cour)
			if rank<topn:
				a=np.asarray([course_list_original[0:topn]])
				b=np.asarray([rec_list[0:topn]])	
				a=a.argsort().argsort()
				b=b.argsort().argsort()
				HR += 1
				if len(a[0])==len(b[0]):
					ndcg_tot=ndcg_tot+1
					NDCG += ndcg_score(a,b)
				else:
					print("error in NDCG calc")
					
				break
	'''
	ndcg_val=ndcg_at_k(rec_list,course_list_original,topn)
			
	NDCG+=ndcg_val
	if best_ndcg<ndcg_val and ndcg_val<1:
		best_ndcg=ndcg_val
		
	#APPLYING THE FIGURED OUT PARTIAL ORDER RELATIONSHIPS
	partial_ordered_rec_list = []
	nval=topn
	for vertex in partial_ordering:
		if vertex in courses_for_minu:
			partial_ordered_rec_list.append(vertex)
			nval -= 1
		if nval == 0:
			break
	if nval!=0:
		for vertex in part_rec_list:
			if vertex not in partial_ordered_rec_list:
				partial_ordered_rec_list.append(vertex)
				nval -= 1
			if nval == 0:
				break	
	#in case the list is too small:
	'''
	partial_ordered_rec_list.extend(rec_list)
	partial_ordered_rec_list=set(partial_ordered_rec_list)
	partial_ordered_rec_list=list(partial_ordered_rec_list)
	partial_ordered_rec_list=partial_ordered_rec_list[0:topn]
	'''	
	
	
	ndcg_val=ndcg_at_k(partial_ordered_rec_list,course_list_original,topn)
	#print("NDCG=",ndcg_val)
	
	NDCG_partial+=ndcg_val
	if best_ndcg_partial<ndcg_val and ndcg_val<1:
		best_ndcg_partial=ndcg_val
		usernumber=unum
		partial_list=partial_ordered_rec_list
		reclist=rec_list
		original_test=course_list_original
		course_list_ult=course_list_ultimate
		#paths=paths_containing_vertices
      		
      		
	set_partial=set(partial_ordered_rec_list)
	try:
		p_partial=len(set_partial & set2)/len(set_partial)
		r_partial=len(set_partial & set2)/len(course_list_original)
		agg_p_partial=agg_p_partial+p_partial
		agg_r_partial=agg_r_partial+r_partial
			
	except:
		#print("division by zero")
		pass
			
	
	#cheking if the courses recommended are actually done by the learner
	set1=set(rec_list)
	#print("set1",set1)
	
	set2=set(course_list_original)
	#print("set2",set2)
	
	if len(set1 & set2)>0:
		HR=HR+1
	if len(set_partial & set2)>0:
		HR_partial=HR_partial+1
	
	try:
		p=len(set1 & set2)/len(set1)
		r=len(set1 & set2)/len(set2)
		agg_p=agg_p+p
		agg_r=agg_r+r
	except:
		#print("division by zero")
		pass
	fn=len(set2.difference(set1))
	if p>best_p:
		best_p=p
		best_r=r
	
	if p_partial>best_p_partial:
		best_p_partial=p_partial
		best_r_partial=r_partial
		usernumber=unum
		partial_list=partial_ordered_rec_list
		reclist=rec_list
		original_test=course_list_original
		course_list_ult=course_list_ultimate
      		
rmseval=math.sqrt(rmseval/total)     		

NDCG=NDCG/total

'''
print("true positives=",tp)
print("false positives=",fp)
print("total=",total)

print("Precision=",precision/total)
print("recall=",recall/total)
print("F1 score=",F1/total)
print("HR=",HR/total)
'''
print("RMSE=",rmseval)

print("results")

print("precision=	",best_p)
print("recall=	",best_r)
print("ndcg",best_ndcg)
print("F1=	", 2*(best_p*best_r+.000001/(best_p+best_r+.000001)))
print("HR=	",HR/total)
print("mndcg=	",NDCG)
print("avgp=",agg_p/total)
print("avgr=",agg_r/total)



print("\n")
print("\n")

print("results_po")
      
p_partial=p_partial/total
r_partial=r_partial/total
NDCG_partial=NDCG_partial/total
print("precision=	",best_p_partial)
print("recall=	",best_r_partial)
print("F1=	", 2*(best_p_partial*best_r_partial+.00001/(best_p_partial+best_r_partial+.000001)))
print("HR=	",HR_partial/total)   
print("ndcg=	",best_ndcg_partial)

print("avgp=",agg_p_partial/total)
print("avgr=",agg_r_partial/total)
print("avgndcg=	",NDCG_partial)
      

print("for results")

print("partial_list:",partial_list)#partially ordered rec list
print("NOSFG reclist:",reclist)#rec list
print("original_test:",original_test)#courses done by the user in test set
print("course_list_ult:",course_list_ult)#whole courses done by the user
#print("paths",paths)
'''

#to get path containing a vertex:
print("path for NSFHG")
print("reclist[0]:",reclist[0])
target_vertex =reclist[0] # Replace 'your_target_vertex' with the actual vertex

# Find paths containing the specified vertex
paths_with_target_vertex = find_paths_containing_vertex(G_transitive, target_vertex)

# Print the result
print(f"Paths containing vertex {target_vertex}:")
for path in paths_with_target_vertex:
    print(path)
    

print("path for PO NSFHG")
print("partial_list[0]:",partial_list[0])
target_vertex =partial_list[0] # Replace 'your_target_vertex' with the actual vertex

# Find paths containing the specified vertex
paths_with_target_vertex = find_paths_containing_vertex(G_transitive, target_vertex)

# Print the result
print(f"Paths containing vertex {target_vertex}:")
for path in paths_with_target_vertex:
    print(path)


'''


		


	



