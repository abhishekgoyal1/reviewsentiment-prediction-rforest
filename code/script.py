from __future__ import division
from math import log
from math import sqrt
import random
import copy
import sys
import pickle
from feature_selection import feature_select1
from normal_feature_selection import feature_select2
from dataset_sampling import sampling

def read_train(feature_list):
	train_list= [0]* 25000
	with open("selected-trainset-indices.txt","r") as f:
		for line in f.readlines():
			value= int(line[:-1])
			train_list[value]=1
	indexcount=-1
	trainset=[]
	with open("aclImdb/train/labeledBow.feat","r") as f:
		for line in f.readlines():
			indexcount=indexcount+1
			if (train_list[indexcount]!=0):
				instance={}
				text=line.split()
				rating = int(text[0])
				positive=0
				if (rating>=7):
					positive=1
				instance['sentiment']= positive
				instance['index']=indexcount
				for i in range(1,len(text)):
					wordindex= int(text[i].split(':',1)[0])
					if (feature_list[wordindex]!=0):
						wordcount = int(text[i].split(':',1)[1])
						instance[wordindex]=wordcount
				trainset.append(instance)

	return trainset

def read_test(feature_list):
	test_list= [0]* 25000
	with open("selected-testset-indices.txt","r") as f:
		for line in f.readlines():
			value= int(line[:-1])
			test_list[value]=1
	indexcount=-1
	testset=[]
	with open("aclImdb/test/labeledBow.feat","r") as f:
		for line in f.readlines():
			indexcount=indexcount+1
			if (test_list[indexcount]!=0):
				instance={}
				text=line.split()
				rating = int(text[0])
				positive=0
				if (rating>=7):
					positive=1
				instance['sentiment']= positive
				instance['index']=indexcount
				for i in range(1,len(text)):
					wordindex= int(text[i].split(':',1)[0])
					if (feature_list[wordindex]!=0):
						wordcount = int(text[i].split(':',1)[1])
						instance[wordindex]=wordcount
				testset.append(instance)

	return testset

def read_validation(feature_list):
	test_list= [0]* 25000
	with open("selected-validationset-indices.txt","r") as f:
		for line in f.readlines():
			value= int(line[:-1])
			test_list[value]=1
	indexcount=-1
	testset=[]
	with open("aclImdb/train/labeledBow.feat","r") as f:
		for line in f.readlines():
			indexcount=indexcount+1
			if (test_list[indexcount]!=0):
				instance={}
				text=line.split()
				rating = int(text[0])
				positive=0
				if (rating>=7):
					positive=1
				instance['sentiment']= positive
				instance['index']=indexcount
				for i in range(1,len(text)):
					wordindex= int(text[i].split(':',1)[0])
					if (feature_list[wordindex]!=0):
						wordcount = int(text[i].split(':',1)[1])
						instance[wordindex]=wordcount
				testset.append(instance)

	return testset

def entropy(x,y):
	total=x+y
	val1=0
	if (x!=0):
		val1= -(x/total)*log(x/total,2)
	val2=0
	if (y!=0):
		val2= -(y/total)*log(y/total,2)
	return val1+val2

def info_gain(dataset, splitpoint):
	countx=0
	county=0
	countx1=0
	county1=0
	countx2=0
	county2=0
	total1=0
	total2=0
	total=len(dataset)
	for i in range(len(dataset)):
		if (dataset[i]['sentiment']==1):
			countx=countx+1
			if (splitpoint in dataset[i]):
				total1=total1+1
				countx1=countx1+1
			else:
				total2=total2+1
				countx2=countx2+1
		else:
			county=county+1
			if (splitpoint in dataset[i]):
				total1=total1+1
				county1=county1+1
			else:
				total2=total2+1
				county2=county2+1

	entropy0= entropy(countx,county)
	entropy1= entropy(countx1,county1)
	entropy2= entropy(countx2,county2)

	return entropy0-entropy1*(total1/total)-entropy2*(total2/total)

def best_feature(dataset, feature_list):
	maxig=0
	bestf=0
	index=0
	for i in range(len(feature_list)):
		ig= info_gain(dataset, feature_list[i])
		if (ig>maxig):
			maxig=ig
			bestf=feature_list[i]
			index=i
	return (bestf,index)


class Node:
	def __init__(self, data):
		self.left = None
		self.right = None
		self.parent= None
		self.data = data

	def nodeCount(self):
		if (self.data==0 or self.data==1):
			return 1
		else:
			lcount=self.left.nodeCount()
			rcount=self.right.nodeCount()
			return lcount+rcount+1

	def leafCount(self):
		if (self.data==0 or self.data==1):
			return 1
		else:
			lcount=self.left.leafCount()
			rcount=self.right.leafCount()
			return lcount+rcount

	def printTree(self):
		print(self.data)
		if self.left:
			self.left.printTree()
		if self.right:
			self.right.printTree()

	def prediction(self, instance):
		if (self.data==0 or self.data==1):
			return self.data
		else:
			if (self.data in instance):
				return self.right.prediction(instance)
			else:
				return self.left.prediction(instance)


	def accuracy(self, dataset):
		count=0
		for i in range(len(dataset)):
			sentimentVal= self.prediction(dataset[i])
			if (sentimentVal==dataset[i]['sentiment']):
				count=count+1
		total= len(dataset)
		return count/total*100

	def pruning(self, root, dataset, testset, count):
		for i in range(count):
			root.prune(root,dataset,testset)
		return

	def depth(self):
		if (self.data==0 or self.data==1):
			return 1
		else:
			return max(self.left.depth()+1,self.right.depth()+1)


	def prune(self, root, dataset, testset):
		if (self.left and self.right):
			if ((self.left.data==0 or self.left.data==1) and (self.right.data==0 or self.right.data==1)):
				initAcc= root.accuracy(testset)
				temp= self.data
				countx=0
				county=0
				for i in range(len(dataset)):
					if (dataset[i]['sentiment']==1):
						countx=countx+1
					else:
						county=county+1
				if (countx>=county):
					self.data=1
				else:
					self.data=0
				removeAcc= root.accuracy(testset)
				if (removeAcc<initAcc):
					self.data=temp
				return
			list1=[]
			list2=[]
			for i in range(len(dataset)):
				if (self.data in dataset[i]):
					list2.append(dataset[i])
				else:
					list1.append(dataset[i])
			self.left.prune(root,list1,testset)
			self.right.prune(root,list2,testset)
		return


					


	def addDTree(self, dataset, feature_list, height):
		if (len(feature_list)==0 or height==0):
			countx=0
			county=0
			for i in range(len(dataset)):
				if (dataset[i]['sentiment']==1):
					countx=countx+1
				else:
					county=county+1
			if (countx>=county):
				self.data=1
			else:
				self.data=0
			return
		bestf_tup=best_feature(dataset,feature_list)
		bestf= bestf_tup[0]
		bestf_index= bestf_tup[1]
		if (bestf==0):
			countx=0
			county=0
			for i in range(len(dataset)):
				if (dataset[i]['sentiment']==1):
					countx=countx+1
				else:
					county=county+1
			if (countx>=county):
				self.data=1
			else:
				self.data=0
			return
		feature_list2= copy.copy(feature_list)
		feature_list2.pop(bestf_index)
		self.data=bestf
		list1=[]
		list2=[]
		for i in range(len(dataset)):
			if (bestf in dataset[i]):
				list2.append(dataset[i])
			else:
				list1.append(dataset[i])
		self.left= Node(-1)
		self.left.parent= self
		self.right= Node(-1)
		self.right.parent = self
		self.left.addDTree(list1,feature_list2,height-1)
		self.right.addDTree(list2,feature_list2,height-1)




def feature_select():
	list1= [0]* 89527
	with open("selected-features-indices.txt","r") as f:
		for line in f.readlines():
			value=int(line[:-1])
			list1[value]=1
	return list1

def feature_select2():
	list1=[]
	with open("selected-features-indices.txt","r") as f:
		for line in f.readlines():
			value=int(line[:-1])
			list1.append(value)
	return list1

def addNoise(dataset, noiseval):
	totalcount= len(dataset)
	noisecount= int(noiseval/100*totalcount)
	noiselist= sorted(random.sample(range(1,1000),noisecount))
	newset= list(dataset)
	for i in range(len(dataset)):
		if ((i+1) in noiselist):
			newset[i]['sentiment']=1-newset[i]['sentiment']
	return newset

def createRandomForest(trainset,feature_list, count):
	m=2000
	list1=[]
	for i in range(count):
		root= Node(-1)
		newfeature_list= sorted(random.sample(feature_list,m))
		root.addDTree(trainset,newfeature_list,1000)
		list1.append(root)
	return list1

def predictRF(treelist, instance):
	count1=0
	count0=0
	for i in range(len(treelist)):
		root= treelist[i]
		value=root.prediction(instance)
		if (value==1):
			count1=count1+1
		else:
			count0=count0+1
	if (count1>=count0):
		return 1
	else:
		return 0

def accuracyRF(treelist, dataset):
	count=0
	for i in range(len(dataset)):
		sentimentVal= predictRF(treelist,dataset[i])
		if (sentimentVal==dataset[i]['sentiment']):
			count=count+1
	total= len(dataset)
	return count/total*100


def main():
	sys.setrecursionlimit(50000)
	if (len(sys.argv)!=2):
		print "Usage: python script.py #no\n where #no is a number from 1-5."
		return
	else:
		choice = int(sys.argv[1])
		if (choice==1):
			print "Initial Preprocessing started."
			print "Selecting 5k features from the vocab list."
			feature_select1()
			print "Done."
			print "Sampling dataset for 1k each of train set, test set and validation set instances."
			sampling()
			print "All done."
		elif (choice==2):
			print "Creating DTree trained on selected 1k trainset."
			feature_mat= feature_select()
			feature_list= feature_select2()
			trainset=read_train(feature_mat)
			testset=read_test(feature_mat)
			root = Node(-1)
			root.addDTree(trainset,feature_list,1000)

			print "Fully Grown Tree Predictions (TrainSet accuracy, TestSet accuracy, LeafCount, NodeCount, TreeDepth):"
			print root.accuracy(trainset), root.accuracy(testset), root.leafCount(), root.nodeCount(), root.depth()
			print "Early Stopping Tree Predictions (Depth of Tree, TrainSet accuracy, TestSet accuracy, LeafCount, NodeCount):"
			root1=Node(-1)
			root1.addDTree(trainset,feature_list,5)
			print "5", root1.accuracy(trainset), root1.accuracy(testset), root1.leafCount(), root1.nodeCount() 
			root1=Node(-1)
			root1.addDTree(trainset,feature_list,10)
			print "10", root1.accuracy(trainset), root1.accuracy(testset), root1.leafCount(), root1.nodeCount() 
			root1=Node(-1)
			root1.addDTree(trainset,feature_list,20)
			print "20", root1.accuracy(trainset), root1.accuracy(testset), root1.leafCount(), root1.nodeCount() 
			root1=Node(-1)
			root1.addDTree(trainset,feature_list,30)
			print "30", root1.accuracy(trainset), root1.accuracy(testset), root1.leafCount(), root1.nodeCount() 
			root1=Node(-1)
			root1.addDTree(trainset,feature_list,40)
			print "40", root1.accuracy(trainset), root1.accuracy(testset), root1.leafCount(), root1.nodeCount() 
			root1=Node(-1)
			root1.addDTree(trainset,feature_list,60)
			print "60", root1.accuracy(trainset), root1.accuracy(testset), root1.leafCount(), root1.nodeCount() 
			root1=Node(-1)
			root1.addDTree(trainset,feature_list,80)
			print "80", root1.accuracy(trainset), root1.accuracy(testset), root1.leafCount(), root1.nodeCount() 
			root1=Node(-1)
			root1.addDTree(trainset,feature_list,100)
			print "100", root1.accuracy(trainset), root1.accuracy(testset), root1.leafCount(), root1.nodeCount() 
			root1=Node(-1)
			root1.addDTree(trainset,feature_list,150)
			print "150", root1.accuracy(trainset), root1.accuracy(testset), root1.leafCount(), root1.nodeCount() 
			print "All Done. Fully Grown Tree stored using pickle module as 'tree-pickledump' file."
			fp = open("tree-pickledump",'wb')
	 		pickle.dump(root,fp)
	 	elif (choice==3):
	 		print "Training the DTree on trainset with added noise."
	 		feature_mat= feature_select()
			feature_list= feature_select2()
			trainset=read_train(feature_mat)
			testset=read_test(feature_mat)
			print "Predictions (NoisePercent Added, TrainSet accuracy, TestSet accuracy, LeafCount, NodeCount, TreeDepth):"
			root = Node(-1)
			root.addDTree(trainset,feature_list,1000)
			print "0.0", root.accuracy(trainset), root.accuracy(testset), root.leafCount(), root.nodeCount(), root.depth()
			newtrainset= addNoise(trainset,0.5)
			root1= Node(-1)
			root1.addDTree(newtrainset,feature_list,1000)
			print "0.5", root1.accuracy(trainset), root1.accuracy(testset), root1.leafCount(), root1.nodeCount(), root1.depth()
			newtrainset= addNoise(trainset,1)
			root1= Node(-1)
			root1.addDTree(newtrainset,feature_list,1000)
			print "1.0", root1.accuracy(trainset), root1.accuracy(testset), root1.leafCount(), root1.nodeCount(), root1.depth()
			newtrainset= addNoise(trainset,5)
			root1= Node(-1)
			root1.addDTree(newtrainset,feature_list,1000)
			print "5.0", root1.accuracy(trainset), root1.accuracy(testset), root1.leafCount(), root1.nodeCount(), root1.depth()
			newtrainset= addNoise(trainset,10)
			root1= Node(-1)
			root1.addDTree(newtrainset,feature_list,1000)
			print "10.0", root1.accuracy(trainset), root1.accuracy(testset), root1.leafCount(), root1.nodeCount(), root1.depth()
			newtrainset= addNoise(trainset,20)
			root1= Node(-1)
			root1.addDTree(newtrainset,feature_list,1000)
			print "20.0", root1.accuracy(trainset), root1.accuracy(testset), root1.leafCount(), root1.nodeCount(), root1.depth()
			print "All Done"
		elif (choice==4):
			print "Fully growing the DTree and then post-pruning."
	 		feature_mat= feature_select()
			feature_list= feature_select2()
			trainset=read_train(feature_mat)
			testset=read_test(feature_mat)
			validationset= read_validation(feature_mat)
			print "Predictions (TrainSet accuracy, TestSet accuracy, LeafCount, NodeCount, TreeDepth):"
			root = Node(-1)
			root.addDTree(trainset,feature_list,1000)
			print "Fully Grown Tree:"
			print root.accuracy(trainset), root.accuracy(testset), root.leafCount(), root.nodeCount(), root.depth()
			root.pruning(root,trainset,validationset,100)
			print "Pruning Result:"
			print root.accuracy(trainset), root.accuracy(testset), root.leafCount(), root.nodeCount(), root.depth()
			fp = open("prunedtree-pickledump",'wb')
	 		pickle.dump(root,fp)
			print "All Done."
		elif (choice==5):
			feature_mat= feature_select()
			feature_list= feature_select2()
			trainset=read_train(feature_mat)
			testset=read_test(feature_mat)
			print "Predictions (Number of RForest Trees, TrainSet accuracy, TestSet accuracy):"
			root = Node(-1)
			root.addDTree(trainset,feature_list,1000)
			print "Fully Grown Single DTree:"
			print "-", root.accuracy(trainset), root.accuracy(testset)
			print "RForest Results:"
			for i in [5,10,15,20,30]:
				rftree= createRandomForest(trainset,feature_list,i)
				print i, accuracyRF(rftree, trainset), accuracyRF(rftree, testset)
				if (i==30):
					fp= open("rforest-pickledump",'wb')
					pickle.dump(rftree,fp)
			print "All Done."
		elif (choice==6):
			feature_mat= feature_select()
			testset=read_test(feature_mat)
			fp= open("tree-pickledump",'r')
			root=pickle.load(fp)
			print "Dtree loaded from pickle dump. Test Set Accuracy: "
			print root.accuracy(testset)
		elif (choice==7):
			feature_mat= feature_select()
			testset=read_test(feature_mat)
			fp= open("prunedtree-pickledump",'r')
			root=pickle.load(fp)
			print "Pruned Dtree loaded from pickle dump. Test Set Accuracy: "
			print root.accuracy(testset)
		elif (choice==8):
			feature_mat= feature_select()
			testset=read_test(feature_mat)
			fp= open("rforest-pickledump",'r')
			rftree=pickle.load(fp)
			print "Random Forest loaded from pickle dump. Test Set Accuracy: "
			print accuracyRF(rftree, testset)
			
	# feature_mat= feature_select()
	# feature_list= feature_select2()
	# trainset=read_train(feature_mat)
	# testset=read_test(feature_mat)
	# validationset= read_validation(feature_mat)

	# rftree= createRandomForest(trainset,feature_list,50)
	# print accuracyRF(rftree, testset)
	# newtrainset= addNoise(trainset,0)
	# root= Node(-1)
	# root.addDTree(trainset,feature_list,1000)
	# print "DTree created suckers!"
	# print root.accuracy(trainset)
	# print root.accuracy(testset)
	# print root.leafCount()
	# print root.nodeCount()
	# root.pruning(root,trainset,validationset,100)
	# print "Dtree pruned :)))"
	# print root.accuracy(trainset)
	# print root.accuracy(testset)
	# print root.leafCount()
	# print root.nodeCount()
	# fp = open("prunedtree-pickledump",'wb')
	# pickle.dump(root,fp)

	# fp= open('prunedtree-pickledump','r')
	# root = pickle.load(fp)
	# print root.accuracy(trainset)
	# print root.accuracy(testset)
	# print root.leafCount()
	# print root.nodeCount()

if __name__=="__main__":
    main()
