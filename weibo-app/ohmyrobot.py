#coding=utf-8

#import urllib2
#import urllib
import time
import os
import subprocess
from weibo import Client

APP_KEY = '********' # app key
APP_SECRET = '*****' # app secret
CALLBACK_URL = 'https://api.weibo.com/oauth2/default.html' # callback url
#AUTH_URL = 'https://api.weibo.com/oauth2/authorize'
USERID = '*******' # userid
PASSWD = '*******' #password
def main():
    """
    获取用户第一条评论,当第一条微博改变时,进行命令执行,
    并在微博评论处返回执行信息
    """

    c = Client(APP_KEY, APP_SECRET, CALLBACK_URL, username=USERID, password=PASSWD)

    print 'Login success'
    print 'Listening...'

    UID = c.get('account/get_uid')['uid']  # 获取用户UID
    last_status = c.get('users/show', uid=UID)['status']  # 获取用户最近微博
    last_id = last_status['id']
    print "last-id: ",last_id
    while True:
        current_status = c.get('users/show', uid=UID)['status']  # 获取用户最近微博
        current_text = current_status['text']
        current_id = current_status['id']
        print time.ctime(),current_text
        if current_id != last_id and current_text[0] == '$':
            if current_text[1] == '@': # 返回执行结果
                current_text = current_text.strip('$@')
                tmp = subprocess.check_output(current_text,shell = True)
                tmp = tmp[:140]
                c.post('comments/create', id=current_id, comment=tmp)
            else:
                current_text = current_text.strip('$@')
                os.system(current_text)
                tmp = "excute success!"
                c.post('comments/create', id=current_id, comment=tmp)
            print tmp
            last_id = current_id  # 更新最近微博id

        time.sleep(5)  # ip限制1000次/小时

if __name__ == '__main__':
    main()
