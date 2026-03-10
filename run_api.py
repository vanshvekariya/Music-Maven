"""Script to run the FastAPI server"""

import uvicorn
from loguru import logger

if __name__ == "__main__":
    logger.info("Starting Music Maven API Server...")
    logger.info("API will be available at: http://localhost:8000")
    logger.info("API Documentation: http://localhost:8000/docs")
    logger.info("Press CTRL+C to stop the server")
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
