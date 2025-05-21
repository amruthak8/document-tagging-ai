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

import logging

app = Flask(__name__)

# Enable CORS for all domains
CORS(app)

# create table
columns_output_table = {
    "document_name": "STRING",
    "document_type": "STRING",
}

create_table_if_not_exists(
    table_name="output_table",
    columns=columns_output_table
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


@app.route('/doc_tagging', methods=['POST'])
def doc_tagging():
    start_time = time.time()
    
    path = f"{config.input_folder}"
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
    app.run(debug=False, port = 5001)
