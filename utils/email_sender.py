import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import os

def send_email(subject, body, to_emails, attachment_path=None):
    from_email = "your_email@example.com"  # 修改为实际邮箱
    password = "your_password"              # 修改为实际密码或授权码
    smtp_server = "smtp.example.com"        # 修改为实际SMTP服务器
    port = 587

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = ','.join(to_emails)
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain', 'utf-8'))

    if attachment_path and os.path.exists(attachment_path):
        with open(attachment_path, "rb") as f:
            part = MIMEApplication(f.read(), Name=os.path.basename(attachment_path))
            part['Content-Disposition'] = f'attachment; filename="{os.path.basename(attachment_path)}"'
            msg.attach(part)

    try:
        with smtplib.SMTP(smtp_server, port) as server:
            server.starttls()
            server.login(from_email, password)
            server.send_message(msg)
    except Exception as e:
        print(f"[ERROR] 邮件发送失败: {e}") 