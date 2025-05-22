from google.cloud import storage

project_id = "aia-aiva-poc-dna-dev-c83dcb"
dataset_id = "transient"

# input_folder = "a_few_files"

output_table_name = "doc_tag_output_table"
agent_reponse_table_name = "agent_response_table"

BUCKET_NAME = "output_aia-aiva-poc-dna-dev-c83dcb"
GEMINI_MODEL="gemini-2.0-flash-001"

# client = storage.Client()
# bucket_name_conf = client.bucket(bucket)

# input_folder = "bpc_ind_type"#"Account_Doc_test"

input_folder = "test"

folder_name = "temp_png"

excel_name = "Audience_data.xlsx"
