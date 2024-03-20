import os
import sys
import time
import warnings

from tqdm import tqdm

warnings.filterwarnings('ignore')  # 忽略一些警告,可以删除
root_path = os.path.abspath(os.path.join(sys.path[0], '..'))
sys.path.append(root_path)

from config.set_config import Config_base
from macls.predict import MAClsPredictor
from macls.data_utils.audio import AudioSegment


class SingleClassifier(object):
    def __init__(self, model_kind, model_path, segment_kind, gpu_id, label_path):
        model_config_path = os.path.join(Config_base.data_witsky_path, "model_conf", "{}.yml".format(model_kind))
        model_path = os.path.join(Config_base.base_path, "models", "{}/best_model/".format(model_path))
        print("加载模型:{}".format(model_path))
        self.gpu_id = gpu_id
        time_start = time.time()
        self.predictor = MAClsPredictor(configs=model_config_path, segment_kind=segment_kind, model_path=model_path, gpu_id=self.gpu_id, label_path=label_path)
        t1_time = time.time()
        print("模型启动消耗时间:{}s".format(str(t1_time - time_start)[0:4]))


    def process_file_glob(self):
        from glob import glob
        all_path = glob("/data_witsky/20230306/*")
        for iter_path in all_path:
            try:
                label, score = self.predictor.predict(audio_data=iter_path)
            except Exception as e:
                print(iter_path)
                print(e)
                exit('111111')
            print(f'\t音频：{iter_path} 的预测结果标签为：{label}，得分：{score}')
        t2_time = time.time()
        # print("模型预测消耗时间:", t2_time - t1_time)
        # print("共消耗时间:", t2_time - time_start)
    
    
    def process_single(self, iter_path, preprocessing=False):
        """预测一个数据"""
        try:
            label, score = self.predictor.predict(audio_data=iter_path, preprocessing=preprocessing)
        except ValueError as E:
            label = "too short"
            score = "too short"
        return label, score


    def predict_pd_batch(self, file_name: str, cut_num=7):
        print("读取数据。。。")
        input_path = os.path.join(Config_base.data_witsky_path, "data_travel", file_name)
        train_pd = Config_base.read_data(os.path.join(Config_base.data_witsky_path, input_path))
        len_data = len(train_pd)
        address_list = list(train_pd["address"])
        print("读取数据完成:{}".format(len_data))
    
        sound_path_list = []
        for iter_index in range(len_data):
            iter_sound_path = address_list[iter_index]
            sound_path_list.append(iter_sound_path)
    
        """创建临时文件夹，存储中间结果"""
        # random_number = np.random.randint(0, 1000)
        # temp_file = os.path.join(Config_base.data_witsky_path, "temp_{}".format(random_number))
        # if Config_base.is_exist(temp_file) is False:
        #     os.makedirs(temp_file)
        # temp_file = os.path.join(Config_base.data_witsky_path, "temp_{}".format(751))
    
        merge_sound_path_list = Config_base.cut_pd(sound_path_list, cut_num)
        result_labels = []
        result_score = []
        for iter_index in tqdm(range(len(merge_sound_path_list))):
            # iter_output_path = os.path.join(temp_file, "{}.pkl".format(iter_index))
            # if Config_base.is_exist(iter_output_path):
            #     print("已经存储:{}".format(iter_output_path))
            #     continue
            try:
                iter_result_labels, iter_result_scores = self.predictor.predict_batch(merge_sound_path_list[iter_index])
            except Exception as E:
                print(E)
                print(merge_sound_path_list[iter_index])
                iter_result_labels = ["error"] * len(merge_sound_path_list[iter_index])
                iter_result_scores = ["error"] * len(merge_sound_path_list[iter_index])
            result_labels.extend(iter_result_labels)
            result_score.extend(iter_result_scores)
            # Config_base.dumps_data([iter_result_labels, iter_result_scores], iter_output_path)
    
        train_pd["result_c2_5"] = result_labels
        train_pd["score_c2_5"] = result_score
        if input_path.endswith("xlsx"):
            output_path = input_path.replace("纯铃音.xlsx", "纯铃音_result.xlsx")
            train_pd.to_excel(output_path, index=False)
        elif input_path.endswith("csv"):
            output_path = input_path.replace(".csv", "_0221_3_result.csv")
            train_pd.to_csv(output_path, index=False)
        else:
            print("文件格式有问题")
        print("完成")

    def predict_pd_travel(self, file_name):
        print("读取数据。。。")
        input_path = os.path.join(Config_base.data_witsky_path, "data_travel", file_name)
        train_pd = Config_base.read_data(input_path)
        address_list = list(train_pd["address"])
        len_data = len(address_list)
        print("读取数据完成:{}".format(len_data))

        result_labels = []
        result_scores = []
        for iter_address in tqdm(address_list):
            label, score = self.process_single(iter_address, True)
            result_labels.append(label)
            result_scores.append(score)
        train_pd["result"] = result_labels
        train_pd["score"] = result_scores
        if input_path.endswith("xlsx"):
            output_path = input_path.replace(".xlsx", "_0220_5_result.xlsx")
            train_pd.to_excel(output_path, index=False)
        elif input_path.endswith("csv"):
            output_path = input_path.replace(".csv", "_0220_5_result.csv")
            train_pd.to_csv(output_path, index=False)
        else:
            print("文件格式有问题")
        print("完成")


class TaskClassifier(object):
    def __init__(self, gpu_id):
        time_start = time.time()
        self.predictor_first = SingleClassifier(model_kind='cam++', segment_kind="random",
                                                model_path='CAMPPlus_Fbank_0221_3',
                                                gpu_id=gpu_id,
                                                label_path=os.path.join(Config_base.label_path))
        self.predictor_second = SingleClassifier(model_kind='cam+2', segment_kind="other",
                                                 model_path='CAMPPlus_Fbank_c2_0221_1',
                                                 gpu_id=gpu_id, label_path=os.path.join(Config_base.label_2_path))
        t1_time = time.time()
        print("模型启动消耗时间:{}s".format(str(t1_time - time_start)[0:4]))

    def process_single(self, input_path, preprocessing=False):
        start_time = time.time()
        single_result_map = {"label": "", "score": "", "isSucc": False, "errMess": "", "speed_time": 0}
        result_label, result_score = self.predictor_first.process_single(input_path, preprocessing)
        if result_label == "BUSY":
            result_label, result_score = self.predictor_second.process_single(input_path, preprocessing)
            if result_label == "ring_tone":
                single_result_map["label"] = "REJECT"
                single_result_map["score"] = result_score
            else:
                single_result_map["label"] = "BUSY"
                single_result_map["score"] = result_score
        elif result_label == "HANGUP":
            print(result_label)
            result_label, result_score = self.predictor_second.process_single(input_path, preprocessing)
            print(result_label, result_score)
            if result_label == "ring_tone":
                single_result_map["label"] = "HANGUP"
                single_result_map["score"] = result_score
            else:
                single_result_map["label"] = "OTHER"
                single_result_map["score"] = result_score
        else:
            single_result_map["label"] = result_label
            single_result_map["score"] = result_score
        end_time = time.time()
        single_result_map["speed_time"] = end_time - start_time
        return single_result_map

    def process_single_memory(self, input_path, preprocessing=False):
        start_time = time.time()
        single_result_map = {"label": "", "score": "", "isSucc": False, "errMess": "", "speed_time": 0}
        audio_segment = AudioSegment.from_file(input_path)

        result_label, result_score = self.predictor_first.process_single(audio_segment, False)
        if result_label == "BUSY":
            result_label, result_score = self.predictor_second.process_single(audio_segment, preprocessing)
            if result_label == "ring_tone":
                single_result_map["label"] = "REJECT"
                single_result_map["score"] = result_score
            else:
                single_result_map["label"] = "BUSY"
                single_result_map["score"] = result_score
        elif result_label == "HANGUP":
            result_label, result_score = self.predictor_second.process_single(audio_segment, preprocessing)
            if result_label == "ring_tone":
                single_result_map["label"] = "HANGUP"
                single_result_map["score"] = result_score
            else:
                single_result_map["label"] = "OTHER"
                single_result_map["score"] = result_score
        else:
            single_result_map["label"] = result_label
            single_result_map["score"] = result_score
        end_time = time.time()
        single_result_map["speed_time"] = end_time - start_time
        return single_result_map


if __name__ == '__main__':
    """初始化模型"""
    # predict_first = SingleClassifier(model_kind='cam++', segment_kind="random", model_path='CAMPPlus_Fbank_0221_3',
    #                                  gpu_id=0,
    #                                  label_path=os.path.join(Config_base.label_path))

    predict_second_1 = SingleClassifier(model_kind='cam+2', segment_kind="other", model_path='CAMPPlus_Fbank_c2_0223_2',
                                      gpu_id=0, label_path=os.path.join(Config_base.label_2_path))
    # tc = TaskClassifier(gpu_id=0)

    '''单条预测'''
    import time
    time_start = time.time()
    for _ in range(100):
        file_path = r"C:\Users\ROG\Desktop\project\audioclassification-pytorch\data_witsky\data_min\对话_1.mp3"
    # file_path = r"/Users/zhuxinquan/Desktop/REJECT.mp3"
    # file_path = os.path.join(Config_base.data_witsky_path, "data_min", "铃音+正在通话中.mp3")
    # file_path = r"/Users/zhuxinquan/Desktop/REJECT.mp3"
        label, score = predict_second_1.process_single(file_path, True)
        print(label)
        print(score)
    print(time.time() - time_start)
    # label, score = predict_second_1.process_single(file_path, True)
    # print(label)
    # print(score)
    # result = tc.process_single(file_path, True)
    # print(result)
    # result = tc.process_single_memory(file_path, True)
    # print(result)


    '''批量预测处理'''

    # predict_first.predict_pd_batch("result_1_4.xlsx")
    # predict_second_2.predict_pd_batch("纯铃音.xlsx")
    # tc.predict_pd_batch("result__total_2_c2.xlsx")


