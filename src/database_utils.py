from google.cloud import bigquery

import logging

project_id = "aia-aiva-poc-dna-dev-c83dcb"
dataset_id = "transient"

client = bigquery.Client(project=project_id)

def create_table_if_not_exists(
    table_name: str,
    columns: dict
):
    print("printing in create_table_if_not_exists")
    """
    Create a BigQuery table inside an existing dataset with dynamic column names.

    Parameters:
        project_id (str): Your GCP project ID
        dataset_id (str): The name of the dataset
        table_name (str): The table to create
        columns (dict): Dictionary of column names and BigQuery types (e.g., {"col1": "STRING", "col2": "INTEGER"})
    """
    print("printing in create_table_if_not_exists")
    
    logging.info
    table_id = f"{project_id}.{dataset_id}.{table_name}"

    schema = [bigquery.SchemaField(name, col_type) for name, col_type in columns.items()]
    table = bigquery.Table(table_id, schema=schema)

    try:
        table = client.create_table(table, exists_ok=True)
        print(f"Table created or already exists: {table_id}")
        
        logging.info(f"Table created or already exists: {table_id}")
        
        logging.info("End of create_table_if_not_exists")
    except Exception as e:
        print(f"Error creating table {table_id}: {e}")
        
        logging.info(f"Error creating table {table_id}: {e}")

def insert_initial_row(table_name: str, row: dict):
    
    print("printing in insert_initial_row")
    
    """
    Inserts a single row into a BigQuery table.

    Args:
        table_name (str): Name of the BigQuery table (without project and dataset).
        row (dict): The row data to insert as a dictionary.

    Returns:
        bool: True if insert succeeded, False otherwise.
    """
    
    try:
        
        logging.info("printing in insert_initial_row")
        table_id = f"{project_id}.{dataset_id}.{table_name}"

        print("table_id: ", table_id)
        print("row: ", row)

        logging.info("table_id: ", table_id)
        logging.info("row: ", row)

        # Get the table metadata
        table = client.get_table(table_id)

        # Extract the column names from the schema
        columns = [field.name for field in table.schema]

        print("columns: ", columns)

        logging.info("columns: ", columns)

        errors = client.insert_rows_json(table_id, [row])
        if errors:
            print("Insert errors:", errors)

            logging.info("Insert errors:", errors)
        else:
            print(f"Inserted row for {row}")

            logging.info(f"Inserted row for {row}")

        logging.info("End of insert_initial_row")
        
    except Exception as e:
        print("Unexpected Error:", e)
        logging.exception("Unexpected Error in insert_initial_row:")
        return False


