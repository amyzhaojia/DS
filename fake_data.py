from faker import Faker
import json
import numpy as np
import pandas as pd
import string

# CallOfferedDiff.json
with open(r"C:\Users\admin\Downloads\calloffer\callofferdata.json") as f:
    data = json.load(f)
file_out = open('./file/CallOfferdata_json_file_20210521.json', "w")
fake_random = np.random.random_integers(1,5)
remove = string.punctuation
table = str.maketrans('ABCDEFGHIGKLMNOPQRSTUVWXYZ', 'MSSTOREONLINESMBWWNAAPGCPQ', remove)
for index1, value1 in data.items():
    for i, val in enumerate(value1):
        val['CallOffered_Diff'] = val['CallOffered_Diff']+fake_random
        val['Upper'] = val['Upper']+fake_random
        val['Lower'] = val['Lower']+fake_random
        val['Internal'] = val['Internal']+fake_random
        val['QueueGroupName'] = val['QueueGroupName'].translate(table)
for index1 in list(data.keys()):
    data[index1.translate(table)] = data.pop(index1)
file_out.write(json.dumps(data))

# # KPIData.json
# with open(r"C:\Users\admin\Downloads\KPIData.json") as f:
#     data = json.load(f)
# file_out = open('./file/KPI_json_file.json', "w")
# fake_random = np.random.random_integers(1,5)
# remove = string.punctuation
# table = str.maketrans('ABCDEFGHIGKLMNOPQRSTUVWXYZ', 'MSSTOREONLINESMBWWNAAPGCPQ', remove)
# for index1, value1 in data.items():
#     for index2, value2 in value1.items():
#         for index3, value3 in value2.items():
#             for m in value3.keys():
#                 if type(value3[m][0])==str:
#                     value3[m] = value3[m]
#                 else:
#                     value3[m] = value3[m] + fake_random
# for index1 in list(data.keys()):
#     data[index1.translate(table)] = data.pop(index1)
# for index1 in list(data.keys()):
#     for index2 in list(data[index1].keys()):
#         data[index1][index2.translate(table)] = data[index1].pop(index2)
# for index1 in list(data.keys()):
#     for index2 in list(data[index1].keys()):
#         for index3 in list(data[index1][index2].keys()):
#             data[index1][index2][index3.translate(table)] = data[index1][index2].pop(index3)
# file_out.write(json.dumps(data))

# # RTMData.json
# with open(r"C:\Users\admin\Downloads\RTMData.json") as f:
#     data = json.load(f)
# file_out = open('./file/RTMData_json_file.json', "w")
# fake_random = np.random.random_integers(1,5)
# remove = string.punctuation
# table = str.maketrans('ABCDEFGHIGKLMNOPQRSTUVWXYZ', 'MSSTOREONLINESMBWWNAAPGCPQ', remove)
# data1 = data
# for value1 in data.values():
#     for value2 in value1.values():
#         for m in value2.keys():
#             if type(value2[m][0])==str:
#                 value2[m] = value2[m]
#             else:
#                 value2[m] = value2[m] + fake_random
#     #     index2 = index2.translate(table)
#     # index1 = index1.translate(table)
# for index1 in list(data.keys()):
#     data[index1.translate(table)] = data.pop(index1)
# for index1 in list(data.keys()):
#     for index2 in list(data[index1].keys()):
#         data[index1][index2.translate(table)] = data[index1].pop(index2)
# file_out.write(json.dumps(data1))

# # SkillRTM.json
# with open(r"C:\Users\admin\Downloads\SkillRTM.json") as f:
#     data = json.load(f)
# file_out = open('./file/SkillRTM_json_file.json', "w")
# fake_random = np.random.random_integers(1,5)
# for m in range(len(data)):
#     for index, value1 in data[m].items():
#         if type(value1)==str:
#             data[m][index] = value1
#         else:
#             data[m][index] = value1 + fake_random
#     remove = string.punctuation
#     table = str.maketrans('ABCDEFGHIGKLMNOPQRSTUVWXYZ', 'MSSTOREONLINESMBWWNAAPGCPQ', remove)
#     data[m]['SkillName'] = data[m]['SkillName'].translate(table)
#     data[m]['LOB'] = data[m]['LOB'].translate(table)
# file_out.write(json.dumps(data))

# # skilldata.json
# with open(r"C:\Users\admin\Downloads\js\js\skilldata.json") as f:
#     data = json.load(f)
# file_out = open('./file/skilldata_json_file.json', "w")
# fake_random = np.random.random_integers(1,5)
# for index1 in data.keys():
#     for m in range(len(data[index1])):
#         for index, value1 in data[index1][m].items():
#             if type(value1)==str:
#                 data[index1][m][index] = value1
#             else:
#                 data[index1][m][index] = value1 + fake_random
#         remove = string.punctuation
#         table = str.maketrans('ABCDEFGHIGKLMNOPQRSTUVWXYZ', 'MSSTOREONLINESMBWWNAAPGCPQ', remove)
#         data[index1][m]['SkillName'] = data[index1][m]['SkillName'].translate(table)
# file_out.write(json.dumps(data))


# calloffer.json
with open(r"C:\Users\admin\Downloads\\calloffer\calloffer.json") as f:
    data = json.load(f)
file_out = open('./file/calloffer_json_file_20210521.json', "w")
remove = string.punctuation
table = str.maketrans('ABCDEFGHIGKLMNOPQRSTUVWXYZ', 'MSSTOREONLINESMBWWNAAPGCPQ', remove)
for m in range(len(data)):
    data[m]['QueueGroupName'] = data[m]['QueueGroupName'].translate(table)
file_out.write(json.dumps(data))


# # skill.json
# with open(r"C:\Users\admin\Downloads\js\js\skill.json") as f:
#     data = json.load(f)
# file_out = open('./file/skill_json_file.json', "w")
# # remove = string.punctuation
# table = str.maketrans('ABCDEFGHIGKLMNOPQRSTUVWXYZ', 'MSSTOREONLINESMBWWNAAPGCPQ')
# for m in range(len(data)):
#     data[m]['SkillName'] = data[m]['SkillName'].translate(table)
# file_out.write(json.dumps(data))


# # skillkpi.json
# with open(r"C:\Users\admin\Downloads\js\js\skillkpi.json") as f:
#     data = json.load(f)
# file_out = open('./file/skillkpi_json_file.json', "w")
# # remove = string.punctuation
# table = str.maketrans('ABCDEFGHIGKLMNOPQRSTUVWXYZ', 'MSSTOREONLINESMBWWNAAPGCPQ')
# for index1,value1 in data.items():
#     for index2, value2 in data[index1].items():
#         for index3, value3 in data[index1][index2].items():
#             data[index1][index2][index3] = value3.translate(table)
# for index1 in list(data.keys()):
#     data[index1.translate(table)] = data.pop(index1)
# for index1 in list(data.keys()):
#     for index2 in list(data[index1].keys()):
#         data[index1][index2.translate(table)] = data[index1].pop(index2)
# for index1 in list(data.keys()):
#     for index2 in list(data[index1].keys()):
#         for index3 in list(data[index1][index2].keys()):
#             data[index1][index2][index3.translate(table)] = data[index1][index2].pop(index3)
# file_out.write(json.dumps(data))


# # skillkpidata.json
# with open(r"C:\Users\admin\Downloads\js_n\js\skillkpidata.json") as f:
#     data = json.load(f)
# file_out = open('./file/skillkpidata_json_file.json', "w")
# # remove = string.punctuation
# table = str.maketrans('ABCDEFGHIGKLMNOPQRSTUVWXYZ', 'MSSTOREONLINESMBWWNAAPGCPQ')
# fake_random = np.random.random_integers(1,5)
# for index1, value1 in data.items():
#     for m in range(len(value1)):
#         for index, value in data[index1][m].items():
#             if type(value)==str:
#                 data[index1][m][index] = value
#             else:
#                 data[index1][m][index] = value+ fake_random
#         if data[index1][m]['Lob']=='':
#             data[index1][m]['Lob'] = 'MLL'
#         else:
#             data[index1][m]['Lob'] = data[index1][m]['Lob'].translate(table)
#         if data[index1][m]['Region'] == '':
#             data[index1][m]['Region'] = 'MLL'
#         else:
#             data[index1][m]['Region'] = data[index1][m]['Region'].translate(table)
#         if data[index1][m]['Language'] == '':
#             data[index1][m]['Language'] = 'MLL'
#         else:
#             data[index1][m]['Language'] = data[index1][m]['Language'].translate(table)
# for index1 in list(data.keys()):
#     data[index1.translate(table)] = data.pop(index1)
# file_out.write(json.dumps(data))

# ## rtmkpi
# import os
# path = r'C:\Users\admin\Downloads\js_n\js\rtmkpi_n'
# fake_random = np.random.random_integers(1,5)
# table = str.maketrans('ABCDEFGHIGKLMNOPQRSTUVWXYZ', 'MSSTOREONLINESMBWWNAAPGCPQ')
# for filename in os.listdir(path):
#     with open(path+'/'+filename) as f:
#         data = json.load(f)
#     for m in range(len(data)):
#         for index, value in data[m].items():
#             if type(value)==int:
#                 data[m][index] = value + fake_random
#             else:
#                 data[m][index] = value
#         # if data[m]['QueueGroupName']==None:
#         #     data[m]['QueueGroupName'] = data[m]['QueueGroupName']
#         # else:
#         #     data[m]['QueueGroupName'] = data[m]['QueueGroupName'].translate(table)
#         data[m]['LOB'] = data[m]['LOB'].translate(table)
#     if data[0]['Region']=='':
#         name_Region = 'MLL'
#     else:
#         name_Region = data[0]['Region']
#     if data[0]['Language'] == '':
#         name_Language = 'MLL'
#     else:
#         name_Language = data[0]['Language']
#     if data[0]['LOB'] == '':
#         name_LOB = 'MLL'
#     else:
#         name_LOB = data[0]['LOB']
#     filename_out = './file/rtmkpi/'+ name_LOB+'-'+name_Region+'-'+name_Language+'.json'
#     filename_out = filename_out.replace(" ","")
#     file_out = open(filename_out, "w")
#     file_out.write(json.dumps(data))
#     # print(os.path.join(path,filename))

# with open(r'C:\Users\admin\Downloads\js_n\js\rtmkpi\rtmkpi.json') as f:
#     data = json.load(f)
# for index1 in list(data.keys()):
#     data[index1.translate(table)] = data.pop(index1)
# file_out = open('./file/rtmkpi/rtmkpi.json', "w")
# file_out.write(json.dumps(data))


# ## rtm
# import os
# path = r'C:\Users\admin\Downloads\js_n\js\rtm_n'
# fake_random = np.random.random_integers(1,5)
# table = str.maketrans('ABCDEFGHIGKLMNOPQRSTUVWXYZ', 'MSSTOREONLINESMBWWNAAPGCPQ')
# for filename in os.listdir(path):
#     with open(path+'/'+filename) as f:
#         data = json.load(f)
#     for m in range(len(data['datas'])):
#         for index, value in data['datas'][m].items():
#             if type(value)==int:
#                 data['datas'][m][index] = value + fake_random
#             else:
#                 data['datas'][m][index] = value
#         if data['datas'][m]['QueueGroupName']==None:
#             data['datas'][m]['QueueGroupName'] = data['datas'][m]['QueueGroupName']
#         else:
#             data['datas'][m]['QueueGroupName'] = data['datas'][m]['QueueGroupName'].translate(table)
#         data['datas'][m]['LOB'] = data['datas'][m]['LOB'].translate(table)
#         data['datas'][m]['QueueGroupKey'] = data['datas'][m]['QueueGroupKey'].translate(table)
#     file_out = open('./file/rtm/'+filename, "w")
#     file_out.write(json.dumps(data))
#     # print(os.path.join(path,filename))
#
# with open(r'C:\Users\admin\Downloads\js\js\rtm\rtm.json') as f:
#     data = json.load(f)
# for index1, value1 in data.items():
#     for index2, value2 in data[index1].items():
#         data[index1][index2] = data[index1][index2].translate(table)
# for index1 in list(data.keys()):
#     data[index1.translate(table)] = data.pop(index1)
# for index1 in list(data.keys()):
#     for index2 in list(data[index1].keys()):
#         data[index1][index2.translate(table)] = data[index1].pop(index2)
# file_out = open('./file/rtm/rtm.json', "w")
# file_out.write(json.dumps(data))
