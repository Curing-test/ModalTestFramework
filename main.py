import os  # 导入操作系统相关模块，后面会用到os.system来运行命令
import sys  # 导入系统相关模块，虽然本文件没用到，但常见于主入口文件
from utils.email_sender import send_email  # 从工具包导入邮件发送函数，用于后续发送评测报告邮件

# 定义一个函数，用于运行pytest测试并生成allure报告
def run_pytest_and_allure():
    # 运行pytest测试用例，并把测试结果保存到reports/allure-results目录
    os.system("pytest --alluredir=reports/allure-results --tb=short")
    # 用allure工具把测试结果生成HTML格式的测试报告，保存在reports/allure-report目录
    os.system("allure generate reports/allure-results -o reports/allure-report --clean")
    # 获取生成的HTML报告的绝对路径
    report_path = os.path.abspath("reports/allure-report/index.html")
    return report_path  # 返回报告路径

# 主流程函数，负责调度评测、生成报告、发送邮件
def main():
    print("[INFO] 开始自动化评测流程...")  # 打印提示信息，告诉用户流程开始了
    report_path = run_pytest_and_allure()  # 执行测试并生成报告，返回报告路径
    print(f"[INFO] Allure报告已生成: {report_path}")  # 打印报告路径，方便用户查看
    # 发送邮件，主题、正文、收件人和附件路径都写死
    send_email(
        subject="自动化评测报告",  # 邮件主题
        body="请查收自动化评测报告，详见附件。",  # 邮件正文
        to_emails=["test@company.com"],  # 收件人列表
        attachment_path=report_path  # 附件路径
    )
    print("[INFO] 邮件已发送。")  # 打印提示信息，告诉用户邮件已发出

# 如果直接运行本文件（而不是被导入），就执行main函数
if __name__ == "__main__":
    main()  # 调用主流程函数 