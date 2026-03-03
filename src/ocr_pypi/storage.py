from commons_pypi.storage import StorageClient
from ocr_pypi.config import settings

def get_storage(bucket: str) -> StorageClient:
    """
    Retorna cliente de storage configurado para o bucket especificado.

    Args:
        bucket: Nome do bucket no R2/S3

    Returns:
        StorageClient configurado
    """
    return StorageClient(
        bucket=bucket,
        endpoint_url=settings.R2_ENDPOINT,
        access_key_id=settings.R2_ACCESS_KEY,
        secret_access_key=settings.R2_SECRET_KEY,
        region_name=settings.R2_REGION,
    )