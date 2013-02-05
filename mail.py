#coding= utf-8
#!/usr/bin/python
import smtplib
import poplib
import sys
import email.mime.text
from email import parser
import string
import email
import arbitrator
import tree

# my test mail
mailUsernameS='articlearbitrator@gmail.com'
mailPasswordS='2xiexing'
fromAddrS = mailUsernameS
#to_addrs=('stevenslxie@gmail.com')

# HOST & PORT
HOST_S = 'smtp.gmail.com'
PORT_S = 25


def sendEmail(to_addrs,content):
	# Create SMTP Object
	smtp = smtplib.SMTP()
	print 'connecting ...'

	# show the debug log
	smtp.set_debuglevel(1)

	# connet
	try:
		smtp.connect(HOST_S,PORT_S)
	except:
		print 'CONNECT ERROR ****'
	# gmail uses ssl
	smtp.starttls()
	# login with username & password
	try:
		print 'loginning ...'
		smtp.login(mailUsernameS,mailPasswordS)
	except:
		print 'LOGIN ERROR ****'
	# fill content with MIMEText's object 
	msg = email.mime.text.MIMEText(content)
	msg['From'] = fromAddrS
	msg['To'] = to_addrs
	msg['Subject']='文章倾向性分析'
#	print msg.as_string()
	smtp.sendmail(fromAddrS,to_addrs,msg.as_string())
	smtp.quit()

def receiveEmail(trees,sams):
	reload(sys)
	sys.setdefaultencoding('gbk')
	host = 'pop.gmail.com'
	username = 'articlearbitrator@gmail.com'
	password = '2xiexing'

	pop_conn = poplib.POP3_SSL(host)
	pop_conn.user(username)
	pop_conn.pass_(password)
	(mailCount,size)=pop_conn.stat()
	print mailCount
	

	#Get messages from server:
	messages = [pop_conn.retr(mailCount-i) for i in range(0, mailCount)]

	# Concat message pieces:
	messages = ["\n".join(mssg[1]) for mssg in messages]

	#Parse message intom an email object:
	messages = [parser.Parser().parsestr(mssg) for mssg in messages]
	for message in messages:
#	message = email.message_from_string(message)
		f = ''
		for part in message.walk():
			if part.get_content_type() == 'text/plain':
				f = f + part.get_payload()
#f=email.Header.decode_header(f)
		print f
#		print message['From']
		reload(arbitrator)
		reload(tree)
#		print message.get_payload()
#	judge = arbitrator.predictSingleArticle(message.get_payload(),trees,sams)
#		sendEmail(message['From'],judge)
			
	pop_conn.quit()


def checkEmail(trees,sams):
	receiveEmail(trees,sams)

