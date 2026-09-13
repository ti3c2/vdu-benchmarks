"""Content-addressed object storage for durable page assets."""

import asyncio
import hashlib
import io
import logging
from dataclasses import dataclass

from minio import Minio
from minio.error import S3Error
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_fixed

from src.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StoredObject:
    bucket: str
    object_key: str
    sha256: str
    mime_type: str
    size_bytes: int
    version_id: str | None = None


class ObjectStore:
    """Use durable bucket/key references; never persist expiring image URLs."""

    def __init__(self, client: Minio | None = None, bucket: str | None = None):
        settings = get_settings()
        self.client = client or Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        self.bucket = bucket or settings.minio_bucket

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(2),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _put_bytes(self, data: bytes, mime_type: str) -> StoredObject:
        digest = hashlib.sha256(data).hexdigest()
        key = f"sha256/{digest[:2]}/{digest}"
        if not self.client.bucket_exists(self.bucket):
            try:
                self.client.make_bucket(self.bucket)
            except S3Error as error:
                if error.code not in {"BucketAlreadyExists", "BucketAlreadyOwnedByYou"}:
                    raise
        try:
            result = self.client.stat_object(self.bucket, key)
        except S3Error as error:
            if error.code not in {"NoSuchKey", "NoSuchObject", "NotFound"}:
                raise
            result = self.client.put_object(
                self.bucket,
                key,
                io.BytesIO(data),
                length=len(data),
                content_type=mime_type,
                metadata={"sha256": digest},
            )
        return StoredObject(
            bucket=self.bucket,
            object_key=key,
            sha256=digest,
            mime_type=mime_type,
            size_bytes=len(data),
            version_id=result.version_id,
        )

    async def put_bytes(self, data: bytes, mime_type: str) -> StoredObject:
        return await asyncio.to_thread(self._put_bytes, data, mime_type)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(2),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _read_bytes(
        self, bucket: str, object_key: str, version_id: str | None
    ) -> bytes:
        response = self.client.get_object(bucket, object_key, version_id=version_id)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    async def read_bytes(
        self, bucket: str, object_key: str, version_id: str | None = None
    ) -> bytes:
        return await asyncio.to_thread(self._read_bytes, bucket, object_key, version_id)

    async def read_asset(self, asset) -> bytes:
        data = await self.read_bytes(asset.bucket, asset.object_key, asset.version_id)
        if hashlib.sha256(data).hexdigest() != asset.sha256:
            raise ValueError(f"Asset {asset.id} failed SHA-256 verification")
        return data
