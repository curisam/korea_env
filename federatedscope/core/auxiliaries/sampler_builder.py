import logging

from federatedscope.core.sampler import UniformSampler, GroupSampler, \
    ResponsivenessRealtedSampler, ClusterUniformSampler,  ClusterAnalSampler     

import os, json

logger = logging.getLogger(__name__)


    
def get_sampler(sample_strategy='uniform', client_num=None, client_info=None, bins=10, config=None):

    """
    This function builds a sampler for sampling clients who should join the \
    aggregation per communication round.

    Args:
        sample_strategy: Sampling strategy of sampler
        client_num: total number of client joining the FL course
        client_info: client information
        bins: size of bins for group sampler

    Returns:
        An instantiated Sampler to sample during aggregation.

    Note:
      The key-value pairs of built-in sampler and source are shown below:
        ===================================  ==============================
        Sampling strategy                    Source
        ===================================  ==============================
        ``uniform``                          ``core.sampler.UniformSampler``
        ``group``                            ``core.sampler.GroupSampler``
        ===================================  ==============================
    """
    if sample_strategy == 'uniform':
        return UniformSampler(client_num=client_num)
    elif sample_strategy == 'responsiveness':
        return ResponsivenessRealtedSampler(client_num=client_num,
                                            client_info=client_info)
    elif sample_strategy == 'group':
        return GroupSampler(client_num=client_num,
                            client_info=client_info,
                            bins=bins)
    

    elif sample_strategy == 'cluster': #gfl_oracle 에서 쓰임.
        # cfg에서 직접 꺼내도 되고, schedule_file을 열어도 됩니다.
        adp = config.llm.adapter

        clusters_1b = adp.clusters
        round_ends  = adp.round_ends
        s_per       = adp.sample_num_per_adapter



        return ClusterUniformSampler(
            client_num=client_num,
            clusters_1b=clusters_1b,
            round_ends=round_ends,                    # ★
            sample_num_per_adapter=s_per
        )
    
    elif sample_strategy == 'anal_cluster':


        rt = getattr(config.llm.adapter, 'cluster_runtime', None)

        if rt is not None:
            if isinstance(rt, dict):
                sched_path = rt.get('schedule_file', None)
            else:
                sched_path = getattr(rt, 'schedule_file', None)

        if sched_path and os.path.exists(sched_path):
            with open(sched_path, 'r') as f:
                info = json.load(f)
            clusters_1b = info.get('clusters_1_based') or info.get('clusters')
        else:
            raise ValueError("고정 그룹핑 초기화 실패: 스케줄/클러스터 정의가 없습니다.")

        # dict: adapter_idx -> [client_ids]
        fixed_groups = {int(aidx): sorted(list(map(int, group)))
                            for aidx, group in enumerate(clusters_1b)}


        return ClusterAnalSampler(
            client_num=client_num, fixed_groups=fixed_groups
        )

    else:
        raise ValueError(
            f"The sample strategy {sample_strategy} has not been provided.")
