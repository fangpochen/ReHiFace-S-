# -- coding: utf-8 --
# @Time : 2021/12/20
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power

import os
import xlrd
from openpyxl import load_workbook
from openpyxl.drawing.image import Image
from cv2box import CVFile, CVImage

image_dir = './data/face_145/'
xlsx_path = './test.xlsx'

# 打开基础信息sheet
wb = load_workbook(xlsx_path)
sheet = wb["基本信息"]


def insert_image(insert_location, image_path):
    img = Image(image_path)
    new_size = (256, 256)
    img.width, img.height = new_size

    sheet[insert_location] = ""
    sheet.add_image(img, insert_location)

    wb.save(xlsx_path)
    print("插入成功!")


pkl_data = CVFile('./data/image_feature_dict.pkl').data
# print(pkl_data)

for k, v in pkl_data.items():
    print("B" + k)
    img = CVImage(v[1]).save('./temp/test{}.jpg'.format(k))
    insert_image("B" + k, './temp/test{}.jpg'.format(k))

wb.save(xlsx_path)
wb.close()
