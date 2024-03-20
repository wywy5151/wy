import os
import sys
import argparse
import functools
import warnings

warnings.filterwarnings('ignore')  # 忽略一些警告,可以删除
root_path = os.path.abspath(os.path.join(sys.path[0], '..'))
sys.path.append(root_path)

from config.set_config import Config_base

from macls.trainer import MAClsTrainer
from macls.utils.utils import add_arguments, print_arguments



parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    os.path.join(Config_base.data_conf_path, "cam++.yml"),        '配置文件')
add_arg("local_rank",       int,    0,                     '多卡训练需要的参数')
add_arg("use_gpu",          bool,   True,                       '是否使用GPU训练')
add_arg('save_model_path',  str,    'models/',                  '模型保存的路径')
add_arg('resume_model',     str,    None,                       '恢复训练，当为None则不使用预训练模型')
add_arg('pretrained_model', str,    None,                       '预训练模型的路径，当为None则不使用预训练模型')
args = parser.parse_args()
print_arguments(args=args)


"""获取训练器"""
# cam++.yml
trainer = MAClsTrainer(configs=args.configs, segment_kind="random", use_gpu=args.use_gpu)
trainer.train(save_model_path=args.save_model_path,
              resume_model=args.resume_model,
              pretrained_model=args.pretrained_model)

# from src.predict_infer import SingleClassifier
# predict_second = SingleClassifier(model_kind='cam++', segment_kind="random", model_path='CAMPPlus_Fbank',
#                                       gpu_id=0, label_path=os.path.join(Config_base.label_2_path))
# predict_second.predict_pd_train("audio_data_train_8.xlsx")

# cam+2.yml
# trainer = MAClsTrainer(configs=args.configs, segment_kind="other", use_gpu=args.use_gpu)
# trainer.train(save_model_path=args.save_model_path,
#               resume_model=args.resume_model,
#               pretrained_model=args.pretrained_model)
# trainer.val()

#
# from src.predict_infer import SingleClassifier
# predict_second = SingleClassifier(model_kind='cam+2', segment_kind="other", model_path='CAMPPlus_Fbank',
#                                       gpu_id=0, label_path=os.path.join(Config_base.label_2_path))
# predict_second.predict_pd_train("audio_data_train_c2_3.xlsx")



"""
pip install  -i https://mirror.baidu.com/pypi/simplepaddlenlp -r requirements.txt
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 train_c2.py
"""