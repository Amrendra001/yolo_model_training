import json
import os
from multiprocessing import Process, Pipe, cpu_count


def s3_cp(source, destination):
    sync_command = f'aws s3 cp "{source}" "{destination}" --quiet'
    os.system(sync_command)


def s3_sync(source, destination):
    sync_command = f'aws s3 sync "{source}" "{destination}" --quiet'
    os.system(sync_command)


def get_best_result(best_result_s3_path, local_data_dir):
    """
        Download best result from s3.
    :return: best run result.
    """
    s3_cp(best_result_s3_path, f'{local_data_dir}/best_result.json')
    with open(f'{local_data_dir}/best_result.json', 'r') as f:
        best_result = json.loads(f.read())
    return best_result