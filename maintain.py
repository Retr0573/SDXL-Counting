import requests
import time
import sys
import os

# sys.path.append('/home/hgh/.local/lib/python3.8/site-packages')
import selenium
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium import webdriver
from time import sleep
from selenium.webdriver import FirefoxOptions


def login():
    opts = FirefoxOptions()
    opts.add_argument("-headless")
    driver = webdriver.Firefox(options=opts)
    print("ok")
    driver.get("https://login.hdu.edu.cn/srun_portal_pc?ac_id=0&theme=pro")
    driver.set_window_size(895, 739)
    driver.implicitly_wait(3)
    try:
        driver.find_element(By.ID, "username").send_keys("232050230")
        driver.find_element(By.ID, "password").send_keys("Wushengqi.547")
        driver.find_element(By.ID, "login-account").click()
    except:
        print("NO!!!!FIND NOTHING!!!")
    finally:
        driver.quit()
        print("WebDriver closed.")
    return 0


# /home/hgh/.local/lib/python3.8/site-packages/selenium/__init__.py


# 定义目标URL，可以是一个外部网站，确保网络请求成功
target_url = "https://www.baidu.com"

# 定义定时发送请求的时间间隔（秒）
interval = 600  # 例如，每隔10分钟发送一次请求

while True:
    try:
        response = requests.get(target_url, timeout=3)
        if response.status_code == 200:
            print("Keep-Alive 请求成功")
            time.sleep(interval)
        else:
            print("Keep-Alive 请求失败")
            login()
    except:
        print("反正有点问题")
        login()
