import logging
from datetime import datetime

def setup_logger():
    
    """
    Configures and initializes a logger that writes logs to a timestamped file.

    This function sets up the Python `logging` module to log messages at the 
    INFO level or higher. The log file is named using the current date and time 
    to ensure uniqueness and traceability. The log format includes timestamps, 
    log levels, and messages.

    Returns:
        str: The filename of the created log file.

    Side Effects:
        - Creates and writes to a new log file in the current working directory.
        - Sets up a global logging configuration for the application.

    Example:
        log_file = setup_logger()
        logging.info("Logging started.")
    """
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"log_{timestamp}.log"

    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    logging.info("Logger initialized.")
    return log_filename