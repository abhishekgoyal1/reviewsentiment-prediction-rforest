def feature_select1():
	index=-1
	wordlist=[]
	with open("aclImdb/imdbEr.txt", "r") as f:
		for line in f.readlines():
			value= float(line[:-1])
			index=index+1
			current=(index,value)
			wordlist.append(current)
			if (index==9999):
				break

	wordlist= sorted(wordlist, key=lambda x: x[1])

	indexlist=[]
	for i in range(500):
		indexlist.append(wordlist[i][0])
	for i in range(500):
		indexlist.append(wordlist[-i][0])


	wordlist=[]
	linecount=-1
	with open("aclImdb/imdbEr.txt", "r") as f:
		for line in f.readlines():
			linecount=linecount+1
			if (linecount>=index):
				value= float(line[:-1])
				current=(index,value)
				index=index+1
				wordlist.append(current)

	wordlist= sorted(wordlist, key=lambda x: x[1])
	for i in range(2000):
		indexlist.append(wordlist[i][0])
	for i in range(2000):
		indexlist.append(wordlist[-i][0])


	indexlist = sorted(indexlist)

	with open("selected-features-indices.txt", "wb") as file:
		for i in range(len(indexlist)):
			file.write(str(indexlist[i]))
			file.write("\n")


if __name__=="__main__":
	feature_select1()