import os
import psutil
def get_current_memory_gb():
    # 获取当前进程内存占用。
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    return info.uss / 1024. / 1024. / 1024.

def get_processes_memory_gb(pid_list):
    # 获取当前进程内存占用。
    return [psutil.Process(p).memory_full_info().uss / 1024. / 1024. / 1024. for p in pid_list]