# 三个未见生成器的数据与预测结果

本目录提供NEAT-POCKET、FLOWR和ResGen三个未见生成器的分子、标签、九方法预测和源数据。三个生成器分别评价，GESA未针对这些集合重新训练或调参。

最常用文件是data/evaluation_molecules.csv：包含三个生成器全部共同有效分子的SMILES、来源、口袋、Retro*结果及九种方法的预测。三个生成器分别独立计算并报告性能。

data/all_scored_molecules.csv额外保留两条因规划超时而未纳入评价的NEAT-POCKET记录。固定SAScore [3,4)子集见data/SAScore_3_to_4_evaluation.csv。原始方向分数和HS方向统一后的HS_score_*列分开保存，后者才是计算当前指标的分数。缺失标签为空，不当作HS。

molecules目录是有效评价分子的原始生成三维结构，SDF属性中已加入预测和标签；raw_molecules保留最终生成尝试的全部原始SDF。索引、质控剔除及对应原因保存在data/generation_record_index.csv和data/structure_audit.csv。

九方法完整AUROC/AP、置信区间及配对差值见metrics目录；图片和对应作图数据在figures目录。数据字典、样本流转、参考论文和代码/权重身份在metadata目录。

原始GESA模型没有重训。本次数据包不包含第三方模型权重和蛋白输入文件。英文README给出了详细方法与字段解释。
