import torch
import numpy as np
from torch.utils.data import Dataset


# huggingface datasets 라이브러리를 import 해야 할 수 있습니다.
try:
    from datasets import Dataset as HFDataset
except ImportError:
    HFDataset = None


class WrapDataset(Dataset):
    """Wrap raw data into pytorch Dataset"""
    def __init__(self, dataset):
        super(WrapDataset, self).__init__()
        self.dataset = dataset
        # dataset이 딕셔너리-유사 객체인지 미리 확인
        # (huggingface Dataset도 이 경우에 해당)
        self.is_dict_like = isinstance(self.dataset, dict) or \
                            (HFDataset is not None and isinstance(self.dataset, HFDataset))

    def __len__(self):
        # 딕셔너리-유사 객체이고 'x' 또는 'y' 키가 있으면, 'y'의 길이 반환
        # LLM 데이터셋은 'input_ids'를 기준으로 길이를 세는 것이 더 안정적일 수 있음
        if self.is_dict_like:
            if 'y' in self.dataset:
                return len(self.dataset['y'])
            elif 'input_ids' in self.dataset: # LLM 데이터셋을 위한 예외 처리
                return len(self.dataset['input_ids'])
            else:
                # 키가 있지만 'y'나 'input_ids'가 없는 경우, 첫 번째 값의 길이 반환
                return len(next(iter(self.dataset.values())))
        else:
            # 딕셔너리 형태가 아니면 (예: list, torch.utils.data.Subset),
            # 객체 자체의 길이를 반환
            return len(self.dataset)

    def __getitem__(self, idx):
        # 딕셔너리-유사 객체일 경우, 각 키에 대해 인덱싱하여 딕셔너리로 반환
        if self.is_dict_like:
            # item 대신 idx를 사용해야 합니다.
            return {key: value[idx] for key, value in self.dataset.items()}
        else:
            # 일반적인 PyTorch Dataset(예: Subset)으로 가정하고 그대로 전달
            return self.dataset[idx]


# class WrapDataset(Dataset):
#     """Wrap raw data into pytorch Dataset

#     Arguments:
#         dataset (dict): raw data dictionary contains "x" and "y"

#     """
#     def __init__(self, dataset):
#         super(WrapDataset, self).__init__()
#         self.dataset = dataset

#     def __getitem__(self, idx):
#         if isinstance(self.dataset["x"][idx], torch.Tensor):
#             return self.dataset["x"][idx], self.dataset["y"][idx]
#         elif isinstance(self.dataset["x"][idx], np.ndarray):
#             return torch.from_numpy(
#                 self.dataset["x"][idx]).float(), torch.from_numpy(
#                     self.dataset["y"][idx]).float()
#         elif isinstance(self.dataset["x"][idx], list):
#             return torch.FloatTensor(self.dataset["x"][idx]), \
#                    torch.FloatTensor(self.dataset["y"][idx])
#         else:
#             raise TypeError

#     def __len__(self):
#         return len(self.dataset["y"])