import urllib #发送请求
import hashlib #加密
import requests
def md5s(strs):
   m = hashlib.md5()
   m.update(strs.encode("utf8")) #进行加密
   return m.hexdigest()

def smsbao(phone, text) :  # 短信宝接口对接

   statusStr = {
       '0' : '短信发送成功',
       '-1' : '参数不全',
       '-2' : '服务器不支持,请确认支持curl或者fsocket,联系您的空间商解决或者更换空间',
       '30' : '密码错误',
       '40' : '账号不存在',
       '41' : '余额不足',
       '42' : '账户已过期',
       '43' : 'IP地址限制',
       '50' : '内容含有敏感词',
       '51' : '手机号码不正确'
   }
   smsapi = "http://api.smsbao.com/"
   # 短信平台账号
   user = 'redmery'
   # 短信平台密码
   password = md5s('060104chen')
   # 要发送的短信内容
   content = str(text)
   # 要发送短信的手机号码
   phone = str(phone)

   data = urllib.parse.urlencode({'u' : user, 'p' : password, 'm' : phone, 'c' : content})
   send_url = smsapi + 'sms?' + data
   response = urllib.request.urlopen(send_url)
   the_page = response.read().decode('utf-8')
   try :
       print(statusStr[the_page])
       return (statusStr[the_page])
   except :
       print('短信发送出现未知错误')
       return '未知错误'


if __name__ == '__main__' :
    # 业务调用部分
    duanx = smsbao(13246857840, '111')  # 调用短信宝接口对接，也就是smsbao方法
