"""
接口调用统计追踪器
自动记录每个 API 端点的调用次数、耗时、错误率
"""

import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from loguru import logger


@dataclass
class EndpointStats:
    """单个端点的统计数据"""
    path: str                          # 端点路径
    method: str                        # HTTP 方法
    total_calls: int = 0               # 总调用次数
    success_calls: int = 0             # 成功次数
    error_calls: int = 0               # 失败次数
    total_time: float = 0.0            # 总耗时
    last_call_time: float = 0.0        # 最近调用时间
    last_duration: float = 0.0         # 最近调用耗时
    max_duration: float = 0.0          # 最大耗时
    min_duration: float = float("inf") # 最小耗时
    recent_durations: List[float] = field(default_factory=list)  # 最近20次耗时

    @property
    def avg_duration(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.total_time / self.total_calls

    @property
    def error_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.error_calls / self.total_calls

    @property
    def recent_avg_duration(self) -> float:
        if not self.recent_durations:
            return 0.0
        return sum(self.recent_durations) / len(self.recent_durations)


class StatsTracker:
    """
    全局统计追踪器 (线程安全)
    用法:
        tracker = StatsTracker()
        tracker.record("/api/v1/ocr/recognize", "POST", 0.5, success=True)
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        # 单例模式
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._stats: Dict[str, EndpointStats] = {}
        self._global_start_time = time.time()
        self._data_lock = threading.RLock()

    def record(self, path: str, method: str, duration: float, success: bool = True):
        """记录一次 API 调用"""
        key = f"{method.upper()} {path}"
        with self._data_lock:
            if key not in self._stats:
                self._stats[key] = EndpointStats(path=path, method=method.upper())

            stat = self._stats[key]
            stat.total_calls += 1
            stat.total_time += duration
            stat.last_call_time = time.time()
            stat.last_duration = duration
            stat.max_duration = max(stat.max_duration, duration)
            stat.min_duration = min(stat.min_duration, duration)

            # 保留最近 20 次耗时
            stat.recent_durations.append(duration)
            if len(stat.recent_durations) > 20:
                stat.recent_durations.pop(0)

            if success:
                stat.success_calls += 1
            else:
                stat.error_calls += 1

    def get_all_stats(self) -> Dict:
        """获取所有统计数据"""
        with self._data_lock:
            uptime = time.time() - self._global_start_time
            endpoints = {}
            total_calls = 0
            total_errors = 0

            for key, stat in self._stats.items():
                endpoints[key] = {
                    "path": stat.path,
                    "method": stat.method,
                    "total_calls": stat.total_calls,
                    "success_calls": stat.success_calls,
                    "error_calls": stat.error_calls,
                    "error_rate": round(stat.error_rate * 100, 2),
                    "avg_duration_ms": round(stat.avg_duration * 1000, 1),
                    "recent_avg_ms": round(stat.recent_avg_duration * 1000, 1),
                    "last_duration_ms": round(stat.last_duration * 1000, 1),
                    "max_duration_ms": round(stat.max_duration * 1000, 1),
                    "min_duration_ms": round(stat.min_duration * 1000, 1) if stat.min_duration != float("inf") else 0,
                    "last_call_ago": round(time.time() - stat.last_call_time) if stat.last_call_time else None,
                }
                total_calls += stat.total_calls
                total_errors += stat.error_calls

            return {
                "uptime_seconds": round(uptime),
                "uptime_human": self._format_uptime(uptime),
                "total_endpoints": len(endpoints),
                "total_calls": total_calls,
                "total_errors": total_errors,
                "overall_error_rate": round(total_errors / total_calls * 100, 2) if total_calls else 0,
                "endpoints": endpoints,
            }

    def get_endpoint_stat(self, path: str, method: str = "POST") -> Optional[dict]:
        """获取单个端点的统计"""
        key = f"{method.upper()} {path}"
        with self._data_lock:
            stat = self._stats.get(key)
            if stat is None:
                return None
            return {
                "path": stat.path,
                "method": stat.method,
                "total_calls": stat.total_calls,
                "success_calls": stat.success_calls,
                "error_calls": stat.error_calls,
                "error_rate": round(stat.error_rate * 100, 2),
                "avg_duration_ms": round(stat.avg_duration * 1000, 1),
                "recent_avg_ms": round(stat.recent_avg_duration * 1000, 1),
                "last_call_ago": round(time.time() - stat.last_call_time) if stat.last_call_time else None,
            }

    def reset(self):
        """重置所有统计"""
        with self._data_lock:
            self._stats.clear()
            self._global_start_time = time.time()

    @staticmethod
    def _format_uptime(seconds: float) -> str:
        """格式化运行时间"""
        seconds = int(seconds)
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            return f"{seconds // 60}m {seconds % 60}s"
        elif seconds < 86400:
            h = seconds // 3600
            m = (seconds % 3600) // 60
            return f"{h}h {m}m"
        else:
            d = seconds // 86400
            h = (seconds % 86400) // 3600
            return f"{d}d {h}h"


# 全局实例
tracker = StatsTracker()
