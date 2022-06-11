import smtplib
import os

def send_email(to_addr: str, gpu_idx: int):
    smtpobj = smtplib.SMTP('smtp-mail.outlook.com', 587)
    smtpobj.ehlo()
    smtpobj.starttls()
    smtpobj.ehlo()
    smtpobj.login('gpuserversender@outlook.jp', os.environ['PASSWORD'])

    from email.mime.text import MIMEText
    from email.utils import formatdate
    msg = MIMEText(f'CUDA: {gpu_idx} terminated')
    msg['Subject'] = 'GPU Server Alart'
    msg['From'] = 'gpuserversender@outlook.jp'
    msg['To'] = to_addr
    msg['Date'] = formatdate()
    smtpobj.sendmail('gpuserversender@outlook.jp', to_addr, msg.as_string())
    smtpobj.close()