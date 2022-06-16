import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_table("./data/logs/multi_experiment_1e_2_test1/multi_experiment_1e_2_test1_s0/progress.txt")
# print(data["AverageJWT"])
# for i in range(len(data)):
#     data["AverageJWT"][i] /= 1 + 2.0 / len(data) * i
#
# z1 = np.polyfit(data['Epoch'], data['AverageEpRet'],4)
# p1 = np.poly1d(z1)
# y = p1(data['Epoch'])

# sns.tsplot(time = 'Epoch',    #时间数据， x轴
#            value = 'AverageJWT', #y轴value
#            # unit="Unit",
#            data = data)
# plt.legend(loc=4).set_draggable(True)
plt.plot(data['TotalEnvInteracts'], data['AverageEpRet'])
# plt.plot(data['TotalEnvInteracts'], y, color="blue")
plt.xlabel("TotalEnvInteracts")
plt.ylabel("AverageReward")
plt.show()