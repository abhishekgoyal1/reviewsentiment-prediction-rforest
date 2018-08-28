import random

def sampling():
	linecount=-1
	positivelist=[]
	negativelist=[]
	with open("aclImdb/train/labeledBow.feat","r") as f:
		for line in f.readlines():
			linecount=linecount+1
			rating = int(line.split(' ',1)[0])
			if (rating>=7):
				positivelist.append(linecount)
			elif (rating<=4):
				negativelist.append(linecount)

	positive_train = sorted(random.sample(positivelist,500))
	negative_train = sorted(random.sample(negativelist,500))

	list1= positive_train + negative_train

	list1= sorted(list1)

	with open("selected-trainset-indices.txt","wb") as file:
		for i in range(len(list1)):
			file.write(str(list1[i]))
			file.write("\n")

	positive_test = sorted(random.sample(range(0,12499),500))
	negative_test = sorted(random.sample(range(12500,24999),500))
	positive_test2 = sorted(random.sample(range(0,12499),500))
	negative_test2 = sorted(random.sample(range(12500,24999),500))

	list2= positive_test + negative_test
	list3= positive_test2 + negative_test2
	list2= sorted(list2)
	list3= sorted(list3)
	with open("selected-testset-indices.txt","wb") as file:
		for i in range(len(list2)):
			file.write(str(list2[i]))
			file.write("\n")

	with open("selected-validationset-indices.txt","wb") as file:
		for i in range(len(list3)):
			file.write(str(list3[i]))
			file.write("\n")


if __name__=="__main__":
	sampling()