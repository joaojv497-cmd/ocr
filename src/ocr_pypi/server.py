import grpc
from concurrent import futures

from Ocr import ocr_pb2_grpc, ocr_pb2
from grpc_reflection.v1alpha import reflection
import logging

from ocr_pypi.config import settings
from ocr_pypi.grpc_server import OCRGrpcServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def create_server() -> grpc.Server:
    """Cria e configura o servidor gRPC"""
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=settings.MAX_WORKERS)
    )

    # Registra serviço OCR
    ocr_pb2_grpc.add_OCRServiceServicer_to_server(
        OCRGrpcServer(),
        server
    )

    # Habilita reflection para debugging/testing
    service_names = (
        ocr_pb2.DESCRIPTOR.services_by_name['OCRService'].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(service_names, server)

    return server


def serve():
    """Inicia o servidor gRPC"""
    server = create_server()

    address = f"[::]:{settings.GRPC_PORT}"
    server.add_insecure_port(address)

    logger.info(f"Servidor gRPC iniciando em {address}")
    server.start()
    logger.info("Servidor gRPC pronto para receber requisições")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Encerrando servidor...")
        server.stop(grace=5)
        logger.info("Servidor encerrado")


if __name__ == "__main__":
    serve()