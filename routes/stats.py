"""
调用统计 API
提供服务运行数据和接口调用统计
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from services.stats import tracker

router = APIRouter()


class StatsResponse(BaseModel):
    """统计响应"""
    uptime_seconds: int
    uptime_human: str
    total_endpoints: int
    total_calls: int
    total_errors: int
    overall_error_rate: float
    endpoints: dict


@router.get("/")
async def get_stats() -> StatsResponse:
    """获取所有接口调用统计"""
    stats = tracker.get_all_stats()
    return StatsResponse(**stats)


@router.get("/reset")
async def reset_stats():
    """重置所有统计"""
    tracker.reset()
    return {"status": "ok", "message": "统计数据已重置"}


@router.get("/endpoint")
async def get_endpoint_stat(path: str, method: str = "POST"):
    """获取单个端点统计"""
    stat = tracker.get_endpoint_stat(path, method)
    if stat is None:
        return {"error": "该端点暂无统计数据"}
    return stat
