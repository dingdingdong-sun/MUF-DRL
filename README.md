# MUF-DRL

## 各个文件用途

| 文件名                     | 用途                                            |
| -------------------------- | ----------------------------------------------- |
| cluster.py                 | 包含cluster与machine类的定义                    |
| multi_HPCSimpickjobs.py    | 包含gym环境代码                                 |
| test_different_schedule.py | 测试不同调度方式性能                            |
| test_my_schedule.py        | 测试PPO-GAIL调度方式性能                        |
| job.py                     | 包含job类与load类，负责日志数据读取生成对应load |
| multi-ppo-pick-jobs.py     | 主程序入口                                      |
| sjf_fair.py                | 数据集分析并生成过滤后样本集                    |

## 使用

python ppo-gail.py --workload "数据集名" --exp_name 实验名 --backfill 0 /1 --epochs 训练总轮次 --trajs 单次采样轨迹数 

## 验证

python test_different_schedule.py --workload "数据集名"

python test_my_schedule.py --workload "数据集名" --pre_trained 1 --trained_model "模型路径"
