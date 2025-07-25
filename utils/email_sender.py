import smtplib  # 导入smtplib库，用于发送邮件
from email.mime.text import MIMEText  # 导入MIMEText，用于构建邮件正文
from email.mime.multipart import MIMEMultipart  # 导入MIMEMultipart，用于构建带附件的邮件
from email.mime.application import MIMEApplication  # 导入MIMEApplication，用于构建附件部分
import os  # 导入os模块，用于文件路径判断

def send_email(subject, body, to_emails, attachment_path=None, cc_emails=None):
    """
    支持SSL的企业邮箱邮件发送，支持附件、抄送、异常处理
    subject: 邮件主题
    body: 邮件正文
    to_emails: 收件人列表
    attachment_path: 附件路径（可选）
    cc_emails: 抄送人列表（可选）
    """
    from_email = "your_email@company.com"  # 发件人邮箱（需替换为实际邮箱）
    password = "your_password"  # 邮箱密码（需替换为实际密码）
    smtp_server = "smtp.company.com"  # 邮箱SMTP服务器（需替换为实际服务器）
    port = 465  # SSL端口，常用465

    msg = MIMEMultipart()  # 创建多部分邮件对象，可以包含正文和附件
    msg['From'] = from_email  # 设置发件人
    msg['To'] = ','.join(to_emails)  # 设置收件人
    if cc_emails:
        msg['Cc'] = ','.join(cc_emails)  # 设置抄送人
    msg['Subject'] = subject  # 设置邮件主题
    msg.attach(MIMEText(body, 'plain', 'utf-8'))  # 添加邮件正文，plain表示纯文本

    # 如果有附件且文件存在，则添加附件
    if attachment_path and os.path.exists(attachment_path):
        with open(attachment_path, "rb") as f:  # 以二进制方式读取附件
            part = MIMEApplication(f.read(), Name=os.path.basename(attachment_path))  # 创建附件对象
            part['Content-Disposition'] = f'attachment; filename="{os.path.basename(attachment_path)}"'  # 设置附件名
            msg.attach(part)  # 添加附件到邮件

    try:
        # 连接SMTP服务器并登录
        with smtplib.SMTP_SSL(smtp_server, port) as server:
            server.login(from_email, password)  # 登录邮箱
            server.send_message(msg)  # 发送邮件
        print("[INFO] 邮件发送成功")  # 打印成功信息
    except Exception as e:
        print(f"[ERROR] 邮件发送失败: {e}")  # 打印错误信息 