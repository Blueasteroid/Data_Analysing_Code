import smtplib
import base64 

def sendemail(from_addr, to_addr_list, cc_addr_list,
              subject, message,
              login, password,
              smtpserver='smtp.gmail.com:587'):
    header  = 'From: %s\n' % from_addr
    header += 'To: %s\n' % ','.join(to_addr_list)
    header += 'Cc: %s\n' % ','.join(cc_addr_list)
    header += 'Subject: %s\n\n' % subject
    message = header + message
 
    server = smtplib.SMTP(smtpserver)
    server.starttls()
    server.login(login,password)
    problems = server.sendmail(from_addr, to_addr_list, message)
    server.quit()
    print("Confirmation email is sent!")
    return problems



if __name__ == "__main__":
    
    #msg_str="Experiment Done!\nWaiting for further instructions...\nJH@KrappLab"

    sendemail(from_addr    = 'jh4209@ic.ac.uk', 
          to_addr_list = ['jh4209@ic.ac.uk'],
          cc_addr_list = [''], 
          subject      = 'I love you!', 
          message      = 'I love you!\nJH@KrappLab', 
          login        = 'jh4209@ic.ac.uk', 
          password     = base64.b64decode(b'not_important').decode("utf-8"),
          smtpserver='smtp.office365.com:587')

    print("Test email is sent!")
