import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus']=False

sns.set_style("whitegrid")
sns.set_context("paper")

# data = pd.DataFrame(columns=["sceduled_method", "user_wait_ratio"])
# print(data)
user_F1 = np.load(r".\user_wait\S_F1.npy")
l =len(user_F1)
data = pd.DataFrame({"sceduled_method": ["F1" for _ in range(l)], "user_wait_ratio": user_F1})
# for w in user_F1:
#     d = pd.DataFrame(["F1", w],index=[str(i)], columns=["sceduled_method", "user_wait_ratio"])
#     i += 1
#     data = data.append(d)
#     print(data)
# print(np.shape(user_F1))
# user_F1 = np.mean(user_F1, axis=1)
user_FCFS = np.load(r".\user_wait\S_FCFS.npy")
l =len(user_FCFS)
data = data.append(pd.DataFrame({"sceduled_method": ["FCFS" for _ in range(l)], "user_wait_ratio": user_FCFS}))
# print(data)
# user_FCFS = np.mean(user_FCFS, axis=1)

# user_MUF_DRL = np.mean(user_MUF_DRL, axis=1)
user_WFP3 = np.load(r".\user_wait\S_WFP3.npy")
l =len(user_WFP3)
data = data.append(pd.DataFrame({"sceduled_method": ["WFP3" for _ in range(l)], "user_wait_ratio": user_WFP3}))
# user_WFP3 = np.mean(user_WFP3, axis=1)
user_SJF = np.load(r".\user_wait\S_SJF.npy")
l =len(user_SJF)
data = data.append(pd.DataFrame({"sceduled_method": ["SJF" for _ in range(l)], "user_wait_ratio": user_SJF}))
# user_SJF = np.mean(user_SJF, axis=1)
user_UNICEP = np.load(r".\user_wait\S_UNICEP.npy")
l =len(user_UNICEP)
data = data.append(pd.DataFrame({"sceduled_method": ["UNICEP" for _ in range(l)], "user_wait_ratio": user_UNICEP}))
# user_UNICEP = np.mean(user_UNICEP, axis=1)
user_MUF_DRL = np.load(r".\user_wait\S-MUF-DRL.npy")
l =len(user_MUF_DRL)
data = data.append(pd.DataFrame({"sceduled_method": ["MUF_DRL" for _ in range(l)], "user_wait_ratio": user_MUF_DRL}))

# dic = {"F1": user_F1, "FCFS": user_FCFS, "MUF_DRL": user_MUF_DRL, "WFP3": user_WFP3, "SJF": user_SJF, "UNICEP": user_UNICEP}
# data = pd.DataFrame.from_dict(dic, orient="index")
#
# print(data)
sns.boxplot(x="sceduled_method", y="user_wait_ratio",data=data)

plt.show()
# data.to_json("./data.json",orient="split")