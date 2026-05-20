import os
import torch
from federatedscope.core.aggregators import Aggregator
import re
import logging


logger = logging.getLogger(__name__)


class AnalysisAggregator(Aggregator):
    """
    Implementation of vanilla FedAvg refer to 'Communication-efficient \
    learning of deep networks from decentralized data' [McMahan et al., 2017] \
    http://proceedings.mlr.press/v54/mcmahan17a.html
    """
    def __init__(self, model=None, device='cpu', config=None):
        super(Aggregator, self).__init__()
        self.model = model
        self.device = device
        self.cfg = config


    def aggregate(self, agg_info): #warmup 라운드 때 적용
        """
        To preform aggregation

        Arguments:
            agg_info (dict): the feedbacks from clients

        Returns:
            dict: the aggregated results
        """

        models = agg_info["client_feedback"] #content= (sample_size, model)들을 담은 리스트
        recover_fun = agg_info['recover_fun'] if (
            'recover_fun' in agg_info and self.cfg.federate.use_ss) else None #None
        avg_model = self._para_weighted_avg(models, recover_fun=recover_fun) #ClientsAvgAggregator와 사실상 동일하게 작동.

        return avg_model
    
    def _para_weighted_avg(self, models, recover_fun=None): 
        """
        여러 클라이언트로부터 받은 모델 dict들을 병합하고, 
        각 Adapter_{adap_idx}를 평균하여 default로 추가하는 함수.

        models: list of (train_size, model_dict)

        """

        # 모델을 병합할 빈 dict를 생성
        merged_model = {}
        

        # 각 클라이언트 모델 합치기
        for train_size, model in models:
            for key, param in model.items():
                if key not in merged_model:
                    merged_model[key] = param
                else:
                    # 이미 같은 key가 존재하면 error 발생
                    raise ValueError(f"Key '{key}' already exists in merged model. Duplicate keys are not allowed.")

        # 각 Adapter_{adap_idx}를 default로 치환하여 평균을 구함
        default_model = {}
        for key, val in merged_model.items():
            if re.search(r'\.Adapter_\d+\.', key): # 정확한 adapter segment만 허용
                # ✅ 정규식으로 'Adapter_X' 패턴 전체를 'default'로 치환
                new_key = re.sub(r'\.Adapter_\d+\.', '.default.', key)
                # 또는 더 간단하게:
                # new_key = key.replace(f"Adapter_{adapter_idx}", "default")로 직접 adapter_idx를 알면...
                # 하지만 여기서는 adapter_idx를 모르므로 정규식 필요
                
                if new_key not in default_model:
                    default_model[new_key] = val
                else:
                    default_model[new_key] = default_model[new_key] + val
            else: # "Adapter_"가 없는 key가 발견되면 에러 발생
                raise ValueError(f"Key '{key}' does not contain 'Adapter_'.")



        # 평균 계산 후, default 모델 추가
        for key, val in default_model.items():
            merged_model[key] = val / len(models)  # 평균을 내기 위해 클라이언트 수로 나눔

        return merged_model



    def update(self, model_parameters): #self.model을 업데이트
        """
        Arguments:
            model_parameters (dict): PyTorch Module object's state_dict.
        """
        self.model.load_state_dict(model_parameters, strict=False)

    def save_model(self, path, cur_round=-1): #당시 (fl라운드, 모든 adapter 모델)을 path에 저장.
        assert self.model is not None


        # 1️⃣ 상위 디렉터리 자동 생성
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # 2️⃣ 파일명에 라운드 번호 붙이기
        base, ext = os.path.splitext(path)                 # ⇒ "checkpoints/tldr_choice_qwen_fedbis", ".ckpt"
        if cur_round >= 0: #이대로 적용.
            path = f"{base}_round_{cur_round}{ext}"         # ⇒ checkpoints/tldr_choice_qwen_fedbis_round50.ckpt


        ckpt = {'cur_round': cur_round, 'model': self.model.state_dict()}
        torch.save(ckpt, path)

    def load_model(self, path):
        assert self.model is not None

        if os.path.exists(path):
            ckpt = torch.load(path, map_location='cpu')
            self.model.load_state_dict(ckpt['model'])
            return ckpt['cur_round']
        else:
            raise ValueError("The file {} does NOT exist".format(path))

