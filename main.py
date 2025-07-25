import os
import sys
from utils.email_sender import send_email

def run_pytest_and_allure():
    # 运行pytest并生成allure原始结果
    os.system("pytest --alluredir=reports/allure-results --clean-alluredir")
    # 生成allure html报告
    os.system("allure generate reports/allure-results -o reports/allure-report --clean")
    report_path = os.path.abspath("reports/allure-report/index.html")
    return report_path

def main():
    print("[INFO] 开始自动化评测流程...")
    report_path = run_pytest_and_allure()
    print(f"[INFO] Allure报告已生成: {report_path}")
    # 邮件发送
    send_email(
        subject="自动化评测报告",
        body="请查收自动化评测报告，详见附件。",
        to_emails=["责任人邮箱@example.com"],
        attachment_path=report_path
    )
    print("[INFO] 邮件已发送。")

if __name__ == "__main__":
    main() 