from flask import Flask, request, jsonify, abort
from flask_cors import CORS  # Import CORS to enable it
import gc
import time
import json, sys
from src.taging_agent import *
from src.workflow import doctagging_workflow
from src.database_utils import create_table_if_not_exists
import config
from logger_config import setup_logger
from datetime import datetime

from src.gcs_utils import GCSFileHandler

import logging

app = Flask(__name__)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Enable CORS for all domains
CORS(app)

# create table
columns_output_table = {
    "document_name": "STRING",
    "document_type": "STRING",
    "industry_segment": "STRING",
    "base_product_code": "STRING",
    "account_focus": "STRING"
}

create_table_if_not_exists(
    table_name = config.output_table_name,
    columns = columns_output_table
)

print("sample table created")

columns_agent_reponse_table = {
    "document_name": "STRING",
    "agent": "STRING",
    "value": "STRING",
    "justification": "STRING"
}

create_table_if_not_exists(
    table_name="agent_response_table",
    columns=columns_agent_reponse_table
)

print("agent response table created")

gcs_handler = GCSFileHandler(config.BUCKET_NAME)

@app.route('/doc_tagging', methods=['POST'])
def doc_tagging():
    
    """
    Handles document tagging requests via a POST API endpoint.

    This endpoint initiates the document tagging process using a predefined 
    LangGraph workflow. It loads input files from a configured input folder, 
    executes the document tagging workflow, and logs relevant information.

    Returns:
        Response (flask.Response): A JSON response indicating the success or 
        failure of the document tagging operation. If successful, it returns 
        a 200 status code. On failure, it returns a 500 status code along 
        with an error message.
    
    Logs:
        - Start and end of the tagging process
        - Processing time in seconds
        - Any errors encountered during the execution

    Raises:
        500 Internal Server Error: If any exception occurs during the 
        document tagging process.
    """
    start_time = time.time()
    
#     if 'input_file' not in request.files:
#         return jsonify({"error": "No file part in the request"}), 400
    
#     file = request.files['input_file']
    
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400
    
    
#     try:
#         # Generate timestamp-based folder name
#         folder_name = f"upload_folder_{timestamp}"

#         # Create the folder in GCS
#         gcs_handler.create_folder(folder_name)

#         # Create the full blob path in GCS
#         destination_blob = f"{folder_name}/{file.filename}"

#         # Upload the file to GCS
#         gcs_handler.upload_file_to_gcs(folder_name, file, destination_blob)

#         logging.info(f"File uploaded to GCS: {destination_blob}")

#     except Exception as e:
#         logging.exception("File upload failed")
#         return jsonify({"status": "error", "message": str(e)}), 500

    
    #path = f"{config.input_folder}"
    
    path = config.input_folder#folder_name
    print("path: ", path)
    
    logging.info("Printing in doc_tagging")
    
    # Create an instance of doc tagging
    state = documenttag(path)
    
    
    try:
        tagging_app = doctagging_workflow(path)
        
        # Run the workflow and return the response
        response = tagging_app.invoke(state,{"aggregate": []})
        
        log_file = setup_logger()
        
        # Free memory
        del state
        del tagging_app
        gc.collect()
        
        end_time = time.time()  # Record end time
        elapsed_time = end_time - start_time  # Calculate the elapsed time
        print(f"Elapsed time: {elapsed_time} seconds")
        
        logging.info(f"Elapsed time: {elapsed_time} seconds")
        
        print(response)
        
        logging.info("End of doc_tagging")
        
        return jsonify({"status": "success"}), 200 #, "Tagged_data": response
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port = 5001)
