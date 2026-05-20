import logging
import numpy as np


import os
import pickle
import torch
import time


from torch.utils.data import random_split, ConcatDataset, Subset, Dataset

import torch.distributed as dist

from federatedscope.core.auxiliaries.splitter_builder import get_splitter
from federatedscope.core.data import ClientData, StandaloneDataDict

logger = logging.getLogger(__name__)


# # --- 환경 변수 기반 헬퍼 함수 ---
# def is_main_process_env():
#     return os.environ.get("LOCAL_RANK", "0") == "0"
# # --------------------------------


def get_rank_info():
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    world = os.environ.get("WORLD_SIZE", os.environ.get("LOCAL_WORLD_SIZE", "1"))
    try: rank = int(rank)
    except Exception: rank = 0
    try: world = int(world)
    except Exception: world = 1
    return rank, world

def is_main_process_env():
    rank, _ = get_rank_info()
    return rank == 0

class BaseDataTranslator:
    """
    Translator is a tool to convert a centralized dataset to \
    ``StandaloneDataDict``, which is the input data of runner.

    Notes:
        The ``Translator`` is consist of several stages:

        Dataset -> ML split (``split_train_val_test()``) -> \
        FL split (``split_to_client()``) -> ``StandaloneDataDict``

    """
    def __init__(self, global_cfg, client_cfgs=None):
        """
        Convert data to `StandaloneDataDict`.

        Args:
            global_cfg: global CfgNode
            client_cfgs: client cfg `Dict`
        """
        #global_cfg 안의 data.splits, federate.client_num, data.splitter 등 설정을 저장
        #get_splitter(global_cfg)로, 클라이언트를 나눌 알고리즘(IID, non-IID, Meta-Split 등)을 준비  

        self.global_cfg = global_cfg
        self.client_cfgs = client_cfgs
        self.splitter = get_splitter(global_cfg) #Metasplitter

    def __call__(self, dataset):
        """
        Args:
            dataset: `torch.utils.data.Dataset`, `List` of (feature, label)
                or split dataset tuple of (train, val, test) or Tuple of
                split dataset with [train, val, test]

        Returns:
            datadict: instance of `StandaloneDataDict`, which is a subclass of
            `dict`.
        """
        datadict = self.split(dataset) #클라이언트별로 데이터를 쪼갠 딕셔너리 반환
        datadict = StandaloneDataDict(datadict, self.global_cfg) #일반적으로 datadict에서 변환되는거 없는 상황.

        return datadict

    def split(self, dataset):
        """
        Perform ML split and FL split.

        Returns:
            dict of ``ClientData`` with client_idx as key to build \
            ``StandaloneDataDict``
        """

        datadict = self.split_to_client(dataset) #FL 관점의 분할: 각 클라이언트 수만큼 균등(또는 지정 방식) 분할
        return datadict

    def split_train_val_test(self, dataset, cfg=None): #ML Split,  이미 (train, val, test) 튜플이 넘어오면 그대로 쓰고, 아니라면 data.splits = [0.9,0.09,0.01] 같은 비율로 랜덤 분할


        """
        Split dataset to train, val, test if not provided.

        Returns:
             List: List of split dataset, like ``[train, val, test]``
        """
        from torch.utils.data import Dataset, Subset

        if cfg is not None:
            splits = cfg.data.splits
        else:
            splits = self.global_cfg.data.splits
        if isinstance(dataset, tuple):
            # No need to split train/val/test for tuple dataset.
            error_msg = 'If dataset is tuple, it must contains ' \
                        'train, valid and test split.'
            assert len(dataset) == len(['train', 'val', 'test']), error_msg
            return [dataset[0], dataset[1], dataset[2]]

        index = np.random.permutation(np.arange(len(dataset)))
        train_size = int(splits[0] * len(dataset))
        val_size = int(splits[1] * len(dataset))

        if isinstance(dataset, Dataset):
            train_dataset = Subset(dataset, index[:train_size])
            val_dataset = Subset(dataset,
                                 index[train_size:train_size + val_size])
            test_dataset = Subset(dataset, index[train_size + val_size:])
        else:
            train_dataset = [dataset[x] for x in index[:train_size]]
            val_dataset = [
                dataset[x] for x in index[train_size:train_size + val_size]
            ]
            test_dataset = [dataset[x] for x in index[train_size + val_size:]]
        return train_dataset, val_dataset, test_dataset


    def split_to_client(self, dataset):
        # ======================= 수정된 부분 시작 =======================
        
        # --- 1. 최종 결과물에 대한 캐싱 로직 ---
        cfg = self.global_cfg

        token_name = cfg.model.type.split('/')[-1].split('@')[0]
        splits_path = os.path.join(getattr(cfg.data, 'splits_path', './final_data_splits'), token_name)

        # splits_path = getattr(cfg.data, 'splits_path', './final_data_splits')
        
        # 전체 딕셔너리를 저장할 단일 캐시 파일
        final_dict_cache_path = os.path.join(splits_path, 'final_datadict.pickle')

        # 동기화를 위한 완료 파일 경로
        completion_file_path = os.path.join(splits_path, 'final_datadict.complete')


        # --- 메인 프로세스가 데이터 생성 및 완료 파일 생성을 책임짐 ---
        if is_main_process_env():
            # 완료 파일이 없으면, 모든 데이터 생성 작업을 수행
            if not os.path.exists(completion_file_path):
                logger.info("Main process: Completion file not found for final_datadict. Generating...")
                
                # 불완전한 파일이 남아있을 수 있으므로, 시작하기 전에 삭제
                if os.path.exists(final_dict_cache_path):
                    os.remove(final_dict_cache_path)
                
                # --- 데이터 생성 및 재구성 로직 (메인 프로세스만 실행) ---
                # 이 로직은 `is_main_process()` 블록 안에 있어야 합니다.
                client_num = cfg.federate.client_num
                # ... (FL-split, 데이터 재구성 로직은 이전 답변과 동일) ...

                split_total = self.splitter(dataset) if len(dataset) > 0 else [[] for _ in range(client_num)]


                data_dict = {}

                # 🔹 여기서 aggregate용 리스트 준비
                agg_train_parts = []
                agg_val_parts = []
                agg_test_parts = []

                for i in range(client_num):
                    client_id = i + 1
                    # 클라이언트 i에게 할당된 초기 데이터셋 조각들을 모음
                    all_client_data = split_total[i] if i < len(split_total) else Subset(dataset, [])
                    total_size = len(all_client_data)
                    # 테스트셋 우선 추출
                    target_test_size = 50

                    generator = torch.Generator().manual_seed(cfg.seed + client_id)
                    remaining_size = total_size - target_test_size
                    #무작위로 섞어서 나눈다.
                    remaining_data, final_test = random_split(
                        all_client_data, [remaining_size, target_test_size], generator=generator
                    )

                    # 나머지로 무작위로 섞어서 Train/Val 재분할
                    remaining_size = len(remaining_data)

                    # 기존 규칙: 5% 또는 최대 200개
                    val_len = 50

                    train_len = remaining_size - val_len
                    final_train, final_val = random_split(
                        remaining_data, [train_len, val_len], generator=generator
                    )

                    # 🔹 aggregate 리스트에 모으기
                    agg_train_parts.append(final_train)
                    agg_val_parts.append(final_val)
                    agg_test_parts.append(final_test)


                    # 클라이언트별 config 설정 (기존 로직)
                    if self.client_cfgs:
                        client_cfg = cfg.clone()
                        client_cfg.merge_from_other_cfg(self.client_cfgs.get(f'client_{client_id}'))
                    else:
                        client_cfg = cfg
                    
                    # 최종 ClientData 객체 생성 및 로그 출력
                    reorganized_cdata = ClientData(client_cfg, final_train, final_val, final_test)
                    logger.info(f"Client {client_id} - Created dataset sizes: "
                                f"Train={len(reorganized_cdata.train_data)}, "
                                f"Val={len(reorganized_cdata.val_data)}, "
                                f"Test={len(reorganized_cdata.test_data)}")
                    
                    data_dict[client_id] = reorganized_cdata


                # 🔹 여기서 최종 aggregate dataset 생성
                final_train_aggregate = ConcatDataset(agg_train_parts)
                final_val_aggregate   = ConcatDataset(agg_val_parts)
                final_test_aggregate  = ConcatDataset(agg_test_parts)

                logger.info(
                    f"Aggregate sizes - Train={len(final_train_aggregate)}, "
                    f"Val={len(final_val_aggregate)}, Test={len(final_test_aggregate)}"
                )



                data_dict[0] = ClientData(cfg, final_train_aggregate, final_val_aggregate, final_test_aggregate)

                # 데이터 파일 저장
                logger.info(f"Main process: Saving final data dict to {final_dict_cache_path}")
                
                os.makedirs(splits_path, exist_ok=True)
                with open(final_dict_cache_path, 'wb') as f:
                    pickle.dump(data_dict, f)
                
                # 모든 작업이 성공적으로 끝나면 완료 파일 생성
                logger.info(f"Main process: Creating completion file at {completion_file_path}")
                with open(completion_file_path, 'w') as f:
                    f.write('done')
            else:
                logger.info("Main process: Completion file found. Skipping generation.")


        # --- 다른 프로세스들은 완료 파일이 생성될 때까지 대기 ---
        else:
            local_rank = os.environ.get("LOCAL_RANK", "?")
            logger.info(f"Process {local_rank}: Waiting for final_datadict completion file...")
            while not os.path.exists(completion_file_path):
                time.sleep(2)
            logger.info(f"Process {local_rank}: Completion file found.")

        # 이제 모든 프로세스는 데이터가 안전하게 준비되었음을 확신하고 로드
        with open(final_dict_cache_path, 'rb') as f:
            data_dict = pickle.load(f)


        # 1) YAML에서 클라이언트 ID 리스트 읽기
        client_ids = getattr(self.global_cfg.aggregator, 'client_range', [])

        # 2) 서버용 데이터 집계 (범위가 없으면 기존 방식 유지)
        agg_train_parts, agg_val_parts, agg_test_parts = [], [], []

        # 리스트가 주어졌을 때만 집계 수행
        if client_ids:
            # 전달된 리스트를 집합(set)으로 변환
            agg_cids = set(int(cid) for cid in client_ids)

            # 해당 ID에 속하는 클라이언트의 데이터만 모음
            for cid in sorted(k for k in data_dict.keys() if k in agg_cids):
                c = data_dict[cid]
                if getattr(c, 'train_data', None):
                    agg_train_parts.append(c.train_data)
                if getattr(c, 'val_data', None):
                    agg_val_parts.append(c.val_data)
                if getattr(c, 'test_data', None):
                    agg_test_parts.append(c.test_data)

        else:  # client_range가 빈 리스트일 경우, 모든 클라이언트에 대해 집계
            for cid in sorted(k for k in data_dict.keys() if k != 0):
                c = data_dict[cid]
                if getattr(c, 'train_data', None) is not None:
                    agg_train_parts.append(c.train_data)
                if getattr(c, 'val_data', None) is not None:
                    agg_val_parts.append(c.val_data)
                if getattr(c, 'test_data', None) is not None:
                    agg_test_parts.append(c.test_data)


        if agg_train_parts or agg_val_parts or agg_test_parts:
            agg_train = ConcatDataset(agg_train_parts) if agg_train_parts else getattr(data_dict[0], 'train_data', None)
            agg_val   = ConcatDataset(agg_val_parts)   if agg_val_parts   else getattr(data_dict[0], 'val_data', None)
            agg_test  = ConcatDataset(agg_test_parts)  if agg_test_parts  else getattr(data_dict[0], 'test_data', None)

        # 서버(키 0)를 aggregate 버전으로 교체
        server_cfg = data_dict[0].cfg if hasattr(data_dict[0], 'cfg') else self.global_cfg
        server_data = ClientData(server_cfg, train=agg_train, val=agg_val, test=agg_test)
        data_dict[0] = server_data



        # === (2) federate.client_num == 1 인 경우: 키 1만 남기고 서버와 동일하게 aggregate 할당 ===
        if self.global_cfg.federate.client_num == 1:
            client_cfg = data_dict[1].cfg if hasattr(data_dict[1], 'cfg') else self.global_cfg
            client_data = ClientData(cfg, train=agg_train, val=agg_val, test=agg_test)
            # 0, 1만 남기고 나머지는 버림
            data_dict = {0: server_data, 1: client_data}


        else: # === (3) 그 외의 경우: 기존 data_dict 유지 ===
            pass


        # ✅ 캐시에서 로드한 경우에도 항상(per rank=0 기본) 요약 출력
        rank, world = get_rank_info()
        force_all = os.environ.get("FS_LOG_SUMMARY_ALL_RANKS", "0") == "1"
        if rank == 0 or force_all:
            def _len_safe(x):
                try: return len(x) if x is not None else 0
                except Exception: return 0

            # 서버(키 0)
            if 0 in data_dict and isinstance(data_dict[0], ClientData):
                s = data_dict[0]
                t = _len_safe(getattr(s, 'train_data', None))
                v = _len_safe(getattr(s, 'val_data', None))
                te = _len_safe(getattr(s, 'test_data', None))
                logger.info(f"[Final Split Summary][loaded][server=0][rank={rank}/{world}] "
                            f"Train={t}, Val={v}, Test={te}, Total={t+v+te}")

            # 클라이언트 1..N
            for cid in sorted(k for k in data_dict.keys() if k != 0):
                c = data_dict[cid]
                t = _len_safe(getattr(c, 'train_data', None))
                v = _len_safe(getattr(c, 'val_data', None))
                te = _len_safe(getattr(c, 'test_data', None))
                logger.info(f"[Final Split Summary][loaded][client={cid}][rank={rank}/{world}] "
                            f"Train={t}, Val={v}, Test={te}, Total={t+v+te}")

        return data_dict




                



       
                    

 
    

# get_data()가 돌려주는 data_dict
"""{
  0: ClientData(server_cfg, train=…, val=…, test=…),
  1: ClientData(client1_cfg, train=…, val=…, test=…),
  2: ClientData(client2_cfg, …),
  …,
  N: ClientData(clientN_cfg, …)
}
Key: 0은 서버, 1~N은 클라이언트

Value: ClientData 인스턴스 (각자의 train/val/test 데이터 보관)

이걸 다시 StandaloneDataDict로 감싸서,
runner 쪽에 넘기면 “각 참가자 ID별 데이터”를 바로 꺼내 쓸 수 있게 되는 구조입니다."""    
