"""
HTTP 请求签名认证 —— 基于 httpx.Auth 的可插拔实现。

用法：
    auth = HmacSignAuth(app_id="xxx", secret_key="yyy")
    http_client = httpx.AsyncClient(auth=auth)
    openai_client = AsyncOpenAI(api_key="...", http_client=http_client)

签名规则（可按实际接口规范修改 _sign 方法）：
    待签字符串 = METHOD\nPATH\nTIMESTAMP\nNONCE\nBODY_MD5
    签名 = HMAC-SHA256(secret_key, 待签字符串).hex()
    请求头:
        X-App-Id:   app_id
        X-Timestamp: unix 秒级时间戳（str）
        X-Nonce:    随机 8 字节 hex
        X-Signature: 签名结果
"""
from __future__ import annotations

import hashlib
import hmac
import os
import time
from typing import Generator

import httpx


class HmacSignAuth(httpx.Auth):
    """
    HMAC-SHA256 签名认证。

    每次请求自动注入签名头，完全无状态，可被多个 httpx.AsyncClient 复用。
    """

    def __init__(self, app_id: str, secret_key: str) -> None:
        self.app_id = app_id
        self._secret = secret_key.encode()

    # httpx.Auth 协议：同步生成器，适用于 sync/async client
    def auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        timestamp = str(int(time.time()))
        nonce = os.urandom(8).hex()

        # 计算 body MD5（空 body 取空串 MD5）
        body_bytes = request.content or b""
        body_md5 = hashlib.md5(body_bytes).hexdigest()

        # 拼接待签字符串
        sign_str = "\n".join(
            [
                request.method.upper(),
                str(request.url.path),
                timestamp,
                nonce,
                body_md5,
            ]
        )

        # HMAC-SHA256 签名
        signature = hmac.new(
            self._secret,
            sign_str.encode(),
            hashlib.sha256,
        ).hexdigest()

        # 注入请求头
        request.headers["X-App-Id"] = self.app_id
        request.headers["X-Timestamp"] = timestamp
        request.headers["X-Nonce"] = nonce
        request.headers["X-Signature"] = signature

        yield request  # 交还给 httpx 发送
