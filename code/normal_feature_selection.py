def feature_select2():
	index=-1
	wordlist=[]
	with open("aclImdb/imdbEr.txt", "r") as f:
		for line in f.readlines():
			value= float(line[:-1])
			index=index+1
			current=(index,value)
			wordlist.append(current)

	wordlist= sorted(wordlist, key=lambda x: x[1])

	indexlist=[]
	for i in range(2500):
		indexlist.append(wordlist[i][0])
	for i in range(2500):
		indexlist.append(wordlist[-i][0])


	indexlist = sorted(indexlist)

	with open("selected-features-indices.txt", "wb") as file:
		for i in range(len(indexlist)):
			file.write(str(indexlist[i]))
			file.write("\n")

if __name__=="__main__":
	feature_select2()