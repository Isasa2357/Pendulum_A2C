import torch
import subprocess
import os
from collections import Counter
from typing import List, Dict, Union

import subprocess
from collections import Counter
from typing import List, Dict, Union

def get_gpu_status() -> List[Dict[str, Union[str, int, float]]]:
    """
    使用可能なすべてのGPUの詳細な状態を取得する関数。
    各GPUに関するリソース使用状況、メモリ、温度、消費電力などの情報を辞書形式で返す。
    さらに、各GPUに割り当てられているプロセス数も取得して返す。

    使用するquery_fieldsはnvidia-smiの項目指定で、取得したいGPU情報を列挙している。
    各フィールドの意味：
        - index: GPUのインデックス番号
        - name: GPU名
        - utilization.gpu: GPUコアの使用率（%）
        - utilization.memory: メモリ使用率（%）
        - memory.total: 総メモリ量（MB）
        - memory.free: 空きメモリ（MB）
        - memory.used: 使用メモリ量（MB）
        - temperature.gpu: GPU温度（℃）
        - power.draw: 現在の消費電力（W）
        - power.limit: 電力上限（W）
        - uuid: GPUのUUID（ユニーク識別子）

    プロセス数は、GPUに割り当てられたプロセス数を取得し、各GPUの情報に追加されます。

    Returns:
        List[Dict[str, Union[str, int, float]]]: 各GPUの情報を含む辞書のリスト。
            各辞書には、GPUのインデックス、名前、リソース使用率、メモリ情報、温度、電力、プロセス数などの情報が含まれます。

    Raises:
        subprocess.CalledProcessError: nvidia-smiコマンドの実行中にエラーが発生した場合。
    """
    try:
        query_fields = [
            'index',
            'name',
            'utilization.gpu',
            'utilization.memory',
            'memory.total',
            'memory.free',
            'memory.used',
            'temperature.gpu',
            'power.draw',
            'power.limit',
            'uuid'
        ]

        # 1回目: GPUごとのステータスを取得（uuidも含む）
        result = subprocess.check_output([
            'nvidia-smi',
            f'--query-gpu={",".join(query_fields)}',
            '--format=csv,nounits,noheader'
        ])
        lines = result.decode('utf-8').strip().split('\n')
        gpu_info_list = []
        uuid_to_index = {}

        # 各GPUの情報を辞書形式で格納
        for line in lines:
            values = [v.strip() for v in line.split(',')]
            gpu_info = dict(zip(query_fields, values))

            # 数値を適切に変換
            for key in query_fields:
                if key != 'uuid':  # uuidは文字列として保持
                    try:
                        gpu_info[key] = float(gpu_info[key]) if '.' in gpu_info[key] else int(gpu_info[key])
                    except ValueError:
                        pass

            uuid_to_index[gpu_info['uuid']] = gpu_info['index']
            gpu_info_list.append(gpu_info)

        # 2回目: プロセスが使っている GPU UUID を取得
        proc_result = subprocess.check_output([
            'nvidia-smi',
            '--query-compute-apps=gpu_uuid',
            '--format=csv,noheader,nounits'
        ])
        proc_lines = proc_result.decode('utf-8').strip().split('\n')
        proc_indices = [uuid_to_index.get(uuid.strip()) for uuid in proc_lines if uuid.strip()]
        proc_count = Counter(i for i in proc_indices if i is not None)

        # 各GPUにプロセス数を追加
        for gpu in gpu_info_list:
            gpu['process.count'] = proc_count.get(gpu['index'], 0)

        return gpu_info_list

    except subprocess.CalledProcessError as e:
        print("Error querying nvidia-smi:", e)
        raise e
    except FileNotFoundError as e:
        print(f'in get_gpu_status: {e}')
        raise e

def select_aprop_gpu(pred_need_memory: int=500, gpu_max_process: int=5, allow_cpu: bool=False) -> str:
    """
    適切なGPUを選択する
    選択されるGPUは"必要なメモリ数が残っている"かつ"割り当てられるプロセス数が空いているものである"
    割り当てに失敗した場合，許可があればCPUを使用する．．
    """

    ### GPUのステータスを取得
    try:
        gpu_status = get_gpu_status()
    except:
        return 'cpu'
    
    ### 判断に必要な情報の抽出
    need_status = [(status['index'], status['memory.free'], status['process.count']) for status in gpu_status if status['memory.free'] > pred_need_memory]

    ### 適切なGPUの選択
    if need_status == []:
        return 'cpu'
    
    aprop_gpu = min(gpu_status, key=lambda x: x['process.count'])
    return f"cuda:{aprop_gpu['index']}"