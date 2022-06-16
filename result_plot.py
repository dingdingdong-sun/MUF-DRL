import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus']=False

sns.set_style("whitegrid")
sns.set_context("paper")

user_F1 = np.load(r".\user_wait\F1.npy")
user_FCFS = np.load(r".\user_wait\FCFS.npy")
user_MUF_DRL = np.load(r".\user_wait\MUF-DRL.npy")
user_WFP3 = np.load(r".\user_wait\WFP3.npy")
user_SJF = np.load(r".\user_wait\SJF.npy")
user_UNICEP = np.load(r".\user_wait\UNICEP.npy")

job_F1 = np.load(r".\job_wait\F1.npy")
job_FCFS = np.load(r".\job_wait\FCFS.npy")
job_MUF_DRL = np.load(r".\job_wait\MUF-DRL.npy")
job_WFP3 = np.load(r".\job_wait\WFP3.npy")
job_SJF = np.load(r".\job_wait\SJF.npy")
job_UNICEP = np.load(r".\job_wait\UNICEP.npy")

sns.color_palette(palette='hls', n_colors=6)
axes = plt.subplot(1, 2, 1)  # 创建一个1行三列的图片

sns.kdeplot(user_FCFS, ax=axes, label="FCFS", cumulative=True)
sns.kdeplot(user_SJF, ax=axes, label="SJF", cumulative=True)
sns.kdeplot(user_WFP3, ax=axes, label="WFP3", cumulative=True)
sns.kdeplot(user_UNICEP, ax=axes, label="UNICEP", cumulative=True)
sns.kdeplot(user_F1, ax=axes, label="F1", cumulative=True)
sns.kdeplot(user_MUF_DRL, ax=axes, label="MUF_DRL", cumulative=True)
plt.xlabel('user wait ratio Cumulative distribution')

axes = plt.subplot(1, 2, 2)
sns.kdeplot(job_FCFS, ax=axes, label="FCFS", cumulative=True)
sns.kdeplot(job_SJF, ax=axes, label="SJF", cumulative=True)
sns.kdeplot(job_WFP3, ax=axes, label="WFP3", cumulative=True)
sns.kdeplot(job_UNICEP, ax=axes, label="UNICEP", cumulative=True)
sns.kdeplot(job_F1, ax=axes, label="F1", cumulative=True)
sns.kdeplot(job_MUF_DRL, ax=axes, label="MUF_DRL", cumulative=True)
plt.xlabel('job wait ratio Cumulative distribution')
plt.show()