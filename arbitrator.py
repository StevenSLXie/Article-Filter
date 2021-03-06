#coding=utf-8 
import urllib2
import sys, os
import re
import string
import jieba
import jieba.analyse
from BeautifulSoup import BeautifulSoup
import csv
from sklearn.ensemble import RandomForestClassifier
from numpy import *
import tree

def encode(s):
    return s.decode('utf-8').encode(sys.stdout.encoding, 'ignore')
 
def getHTML(url):
	req = urllib2.Request(url)
	response = urllib2.urlopen(req, timeout=3000)
	return BeautifulSoup(response,fromEncoding="gb18030")
	

def visible(element):
    '''抓取可见的文本元素'''
    if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
        return False
    elif re.match('<!--.*-->', str(element)):
        return False
    elif element == u'\xa0':
        return False
 
    return True
 
def delReturn(element):
    '''删除元素内的换行'''
    return re.sub('(?<!^)\n+(?!$)', ' ', str(element)).decode('utf-8')
 
def validFilename(filename):
    # windows
    return re.sub('[\/:*?<>"|\xa0]', '', filename)
 
def writeToFile(text, filename, dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print encode('保存到目录'), dirname
 
    filename = validFilename(filename)
    print encode('保存文章'), filename
 
    path = os.path.join(dirname, filename)
    if not os.path.exists(path):
        f = open(path, 'w')
        f.write(text)
        f.close()
    else:
        print filename, encode('已经存在')

def formatContent(url, title=' ',idStr='wrong',folderName='wrong'):
	page = getHTML(url)
#	print page
	content = page.find(id=idStr)    # change 24-01-2013 from find to findAll
#	print content
	art_id = url[-4:]
	blog_name = folderName
	temp_data = filter(visible, content.findAll(text=True)) # 去掉不可见元素
	temp_data = ''.join(map(delReturn, temp_data)) # 删除元素内的换行符
	temp_data = temp_data.strip() # 删除文章首尾的空行
	temp_data = re.sub('\n{2,}', '\n\n', temp_data) # 删除文章内过多的空行
 
    # 输出到文件
    # 编码问题
#temp_data = '本文地址:'.decode('utf-8') + url + '\n\n' + temp_data
	op_text = temp_data.encode('utf-8')
	op_file = str(title)+'.txt'
 
	writeToFile(op_text, op_file, blog_name)
 
def articlelist(url):
	k = 81
	pageNum = 13;
	urlBase = url[:-1]
	for i in range(pageNum,6+pageNum):
		cate = getHTML(urlBase+str(i))
		art = cate.findAll('a',href=re.compile("^portal.php\?mod=view"))
		for j in range(1,len(art)):
			k = k+1
			formatContent('http://www.impencil.org/'+str(art[j]['href']),k,"article_content","Positive")
#	return art


def blog_dld(articles):
    if not isinstance(articles, dict):
        return False
 
    print encode('开始下载文章')
    for art_title, art_href in articles.items():
        formatContent(art_href, art_title)

def analyzeArticle(addr,artType):
	addr = '/Users/xingmanjie/Applications/Python/Recommend/'+artType+'/'+str(addr)+'.txt'
	f = open(addr).read().replace('\n','')
	'''
	f = f.replace('我们','')
	f = f.replace('&nbsp','')
	f = f.replace('他们','')
	f = f.replace('的','')
	f = f.replace('这','')
	f = f.replace('那','')
	f = f.replace('了','')
	f = f.replace('就','')
	f = f.replace('会','')
	f = f.replace('可能','')
	'''
	return f

def createTrainingData2(num=35,artType='wrong'):
	train = ''
	for i in range(num+1):
		temp = ''
		temp = analyzeArticle(i+1,artType)
		if artType == 'Positive':
			target = '\t1\n'
		else:
			target = '\t0\n'
		temp = temp + target 
#train.append(temp)
		train = train + temp
#	print train
	file('train'+artType+'.txt','w').write(train)


def formKeyWords(artNum=35,artType = 'wrong'):
	h = []
	temp = ""
	for i in range(artNum+1):
		f =  analyzeArticle(i+1,artType)
		temp = temp + f

	tags = jieba.analyse.extract_tags(temp,topK=600)
	print ",".join(tags)
	return tags

def saveKeyWords(fileName):
	result = formKeyWords()
	'''
	with open(fileName, 'wb') as csvfile:
		writer = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
		for i in result:
			writer.writerow(i)
	'''

def mergeTwoItems(a,b):
	mer = []
	for i in range(len(a)):
#		a[i]= a[i].encode('utf-8')
		mer.append(a[i])
	
	for i in range(len(b)):
		flag = 0
#		b[i]= b[i].encode('utf-8')
		for j in range(len(a)):
			if b[i]==a[j]:
				flag = 1
				break
		if flag == 0:
			mer.append(b[i])
	return mer

def encodeToUtf(a):
	new = []
	for i in range(len(a)):
		a[i] = a[i].encode('utf-8')
		new.append(a[i])
	return new

def mergeKeyWords():

	neg=formKeyWords(35,'Negative')
	pos=formKeyWords(35,'Positive')
	sci=formKeyWords(35,'Science')

	neg = encodeToUtf(neg)
	pos = encodeToUtf(pos)
	sci = encodeToUtf(sci)
	
	key = mergeTwoItems(neg,pos)
	key = mergeTwoItems(sci,key)

	return key


def createArtVector(tags,addr,artType,source='None'):
# transform an article into a vector indicating whether it hit the key words or not
#	for i in range(len(tags)):
#		tags[i] = tags[i].encode('utf-8')
	if source == 'None':
		addr1 = '/Users/xingmanjie/Applications/Python/Recommend/'+artType+'/'+str(addr)+'.txt'
		f = open(addr1).read()
	else:
		f = str(addr)
	f = jieba.analyse.extract_tags(f,topK=400)
#	print f

		
	for i in range(len(f)):
		f[i] = f[i].encode('utf-8')

	v = []
	for i in range(len(tags)):
		flag = 0
		for j in range(len(f)):
			if f[j]==tags[i]:
				flag = 1
				v.append(1)
				print f[j]
				break
		if flag == 0:
			v.append(0)
	if artType == 'Positive':
		v.append(2)  # y, indicate that is positive
	elif artType == 'Negative':
		v.append(0)
	elif artType == 'Science':
		v.append(1)
	return v

def createTrainingData(num = 35,artType = 'wrong'):
	train = []
	for i in range(num):
		tags = mergeKeyWords()
		train.append(createArtVector(tags,i+1,artType))
	
	with open(artType+'.csv', 'wb') as csvfile:
		writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
		for i in train:
			writer.writerow(i)


def createAllTraining(num=35):
	train = []
	for i in range(num):
		tags = mergeKeyWords()
		train.append(createArtVector(tags,i+1,'Positive'))

	for i in range(num):
		tags = mergeKeyWords()
		train.append(createArtVector(tags,i+1,'Negative'))

	for i in range(num):
		tags = mergeKeyWords()
		train.append(createArtVector(tags,i+1,'Science'))

	with open('Train.csv', 'wb') as csvfile:
		writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
		for i in train:
			writer.writerow(i)
	

def cleanTrainingData(fileName = 'Train.csv'):
	data = loadDataSet('Train.csv')
	target = [x[-1] for x in data[0:70]]
	train = [x[0:-1] for x in data[0:70]]
	test = [x[0:-1] for x in data[60:68]]

	target = strToInt(target)
	train = strToInt(train)
	test = strToInt(test)
	
	return target,train,test


def randomForest():	
	target,train,test = cleanTrainingData()
#	print len(train[1])
	rf = RandomForestClassifier(n_estimators=300,n_jobs=2)
	rf.fit(train,target)
	predicted_probs = [x[1] for x in rf.predict_proba(test)]

	print	predicted_probs
#	print target[3]
	return rf

def loadTreeData(fileName):
	data = loadDataSet(fileName,str=',')
	data = strToInt(data)
	data = array(data)
	data = asmatrix(data)

	return data


def predictTestData():
	target,train,test = cleanTrainingData()	
	data = loadDataSet('Test.csv')
	data = [x for x in data]
	data = strToInt(data)
	print len(data[2])

	rf = RandomForestClassifier(n_estimators=2000,n_jobs=2)
	rf.fit(train,target)

	predicted_probs = [x[1] for x in rf.predict_proba(data)]
	
	for i in range(len(predicted_probs)):
		print i+1,predicted_probs[i]

def predictSingleArticle(art,trees,sams):
	tags = mergeKeyWords()
	test = createArtVector(tags,art,'Test','None')
	print test
	result = tree.batchPredict(test,trees,sams)
	if result == 0:
		return '激进左派；共产主义；支持大政府'
	elif result == 1:
		return '温和左派；福利主义；无特别明显经济观点'
	elif result == 2:
		return '无明显经济观点；可能为科普文章等'
	elif result == 3:
		return '中间偏右；基本支持市场经济；保守主义；强调资本主义精神等'
	elif result == 4:
		return '经济意义上的右派观点；支持市场经济；强调个人权利；保守主义'

	


def strToInt(ori):
	modi = []
	for x in ori:
		temp = []
		for i in range(len(x)):
			temp.append(int(x[i]))
		modi.append(temp)

	return modi

def loadDataSet(fileName,str=","):      #general function to parse tab -delimited floats
	dataMat = []                #assume last column is target value
	with open(fileName) as fr:
		for line in fr:
			curLine = line.strip().split(str)
#			curLine = map(float,curLine) #map all elements to float()
			dataMat.append(curLine)
	return dataMat

#if __name__ == '__main__':
def init():
	sel = raw_input(encode('你要下载的是(1)全部文章还是(2)单篇文章（3）右派（4）左派: '))
 
	if sel == '1':
		articlelist_url = raw_input(encode('请输入博客文章目录链接: '))
		articles = articlelist(articlelist_url)
		blog_dld(articles)
	elif sel == '2':
		article_url = raw_input(encode('请输入博客文章链接: '))
		formatContent(article_url)
	elif sel == '3':
		article_url = raw_input(encode('请输入博客文章链接: '))
		h=articlelist(article_url)
	else:
		nfzmList()
		

