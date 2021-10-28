

pf_1layer_config = {
    "type":'PF_1layer',
    "dims":64,
    "pf_path":None,  # 预训练模型, 若存在, 则加载
    "model_path":None,  # 分类模型路径, 若存在, 则加载
    "use_pf":True,  # 是否使用pointfield, 不使用则输入直接传入model
    "freeze_pf":True,  # 是否冻结pointfield
    "freeze_model":True,  # 是否冻结分类模型
    "weights":[1, 1],  # tri_loss 和 reg_loss 的权重
    "initial":"normal",
    "loss":{
        "selector":"hardest_negtive",
        "margin":0.3
    },
    "resume":False,  # 是否恢复训练
}

pf_tnet_config = {
    "type":'PF_Tnet',
    "dims":64,
    "pf_path":None,  # 预训练模型, 若存在且resume=False, 则加载
    "model_path":r'D:\work\PointClassifier\best_models\best_dgcnn.t7',  # 分类模型路径, 若存在且resume=False, 则加载
    "use_pf":True,  # 是否使用pointfield, 不使用则输入直接传入model
    "freeze_pf":False,  # 是否冻结pointfield
    "freeze_model":True,  # 是否冻结分类模型
    "weights":[1, 1, 0.1],  # tri_loss, reg_loss, cls_loss 的权重
    "initial":"zero", # 初始化为0
    "loss":{
        "selector":"random_negtive",
        "margin":0.3
    },
    "resume":False,  # 是否恢复训练, 若是则不在创建PFWithModel时加载预训练模型, 在Trainer中加载
}

pf_config = {
    "type":'PF_1layer',
    "dims":64,
    "pf_path":None,  # 预训练模型, 若存在且resume=False, 则加载
    "model_path":r'D:\work\PointClassifier\best_models\best_dgcnn.t7',  # 分类模型路径, 若存在且resume=False, 则加载
    "use_pf":True,  # 是否使用pointfield, 不使用则输入直接传入model
    "freeze_pf":False,  # 是否冻结pointfield
    "freeze_model":True,  # 是否冻结分类模型
    "weights":[1, 1, 0.1],  # tri_loss, reg_loss, cls_loss 的权重
    "initial":"zero",  # 初始化设置
    "loss":{
        "selector":"hardest_negtive",
        "margin":0.3
    },
    "resume":False,  # 是否恢复训练, 若是则不在创建PFWithModel时加载预训练模型, 在Trainer中加载
}
