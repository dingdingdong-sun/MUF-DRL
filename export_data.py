import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus']=False
#
# sns.set_style("whitegrid")
# sns.set_context("paper")

# data = pd.DataFrame(columns=["sceduled_method", "user_wait_ratio"])
# print(data)
data = {}
user_F1 = np.load(r".\user_wait\F1_backfill.npy")
data["F1"] = user_F1.tolist()
# print(len(data["F1"]))
# data = pd.DataFrame({"sceduled_method": ["F1" for _ in range(l)], "user_wait_ratio": user_F1})
# for w in user_F1:
#     d = pd.DataFrame(["F1", w],index=[str(i)], columns=["sceduled_method", "user_wait_ratio"])
#     i += 1
#     data = data.append(d)
#     print(data)
# print(np.shape(user_F1))
# user_F1 = np.mean(user_F1, axis=1)
user_FCFS = np.load(r".\user_wait\FCFS_backfill.npy")
data["FCFS"] = user_FCFS.tolist()
# data = data.append(pd.DataFrame({"sceduled_method": ["FCFS" for _ in range(l)], "user_wait_ratio": user_FCFS}))
# print(data)
# user_FCFS = np.mean(user_FCFS, axis=1)

# user_MUF_DRL = np.mean(user_MUF_DRL, axis=1)
user_WFP3 = np.load(r".\user_wait\WFP3_backfill.npy")
data["WFP3"] = user_WFP3.tolist()
# user_WFP3 = np.mean(user_WFP3, axis=1)
user_SJF = np.load(r".\user_wait\SJF_backfill.npy")
data["SJF"] = user_SJF.tolist()
# user_SJF = np.mean(user_SJF, axis=1)
user_UNICEP = np.load(r".\user_wait\UNICEP_backfill.npy")
data["UNICEP"] = user_UNICEP.tolist()
# user_UNICEP = np.mean(user_UNICEP, axis=1)
user_MUF_DRL = np.load(r".\user_wait\MUF-DRL_backfill.npy")
data["MUF_DRL"] = user_MUF_DRL.tolist()

with open("data.json", "w") as f:
    # data = json.dumps(data)
    json.dump(data, f)

# dic = {"F1": user_F1, "FCFS": user_FCFS, "MUF_DRL": user_MUF_DRL, "WFP3": user_WFP3, "SJF": user_SJF, "UNICEP": user_UNICEP}
# data = pd.DataFrame.from_dict(dic, orient="index")
#
# print(data)
# sns.boxplot(x="sceduled_method", y="user_wait_ratio",data=data)
#
# plt.show()
# data.to_json("./data.json",orient="split")