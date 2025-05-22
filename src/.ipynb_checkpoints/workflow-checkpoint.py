import concurrent.futures
from langgraph.graph import StateGraph, START, END

import logging

from src.taging_agent import documenttag, documenttag_segment

def doctagging_workflow(path) -> StateGraph:
    
    """
    Constructs and compiles the LangGraph workflow for document tagging.

    This function initializes the document tagging segment using the provided 
    input path and builds a LangGraph `StateGraph` workflow. The workflow includes 
    multiple tagging components (nodes) such as document reader, base product code tagger, 
    industry segment tagger, document type identifier, and account focus identifier. 
    All nodes eventually feed into a post-processing step before ending the workflow.

    Args:
        path (str): The file path to the folder containing the input documents 
                    to be processed.

    Returns:
        StateGraph: A compiled LangGraph `StateGraph` instance that represents 
        the entire document tagging workflow.

    Workflow Nodes:
        - document_reader: Extracts raw data from input documents.
        - base_product_code: Tags the base product code from extracted data.
        - industry_segment: Tags the industry segment.
        - resulting_doc_type: Identifies the type of the document.
        - resulting_account_focus: Determines account focus.
        - post_processing: Aggregates and finalizes the tagging results.

    Raises:
        Any exceptions that occur during node creation or graph compilation 
        will propagate up the call stack.

    Logs:
        - Each step of the workflow setup.
        - Start and end of the workflow construction process.
    """
    print("Printing in doctagging_workflow")
    
    logging.info("Printing in doctagging_workflow")
    
    try:
    
        # Create an instance of Document Tagging
        doctag_inst = documenttag_segment(path)

        print("Printing in Langgraph")    
        logging.info("Printing in Langgraph") 
        # Define the LangGraph Workflow
        doctag_workflow = StateGraph(doctag_inst)
        doctag_workflow.add_node("document_reader", doctag_inst.extract_data)
        doctag_workflow.add_node("base_product_code", doctag_inst.base_product_code_tag)
        doctag_workflow.add_node("industry_segment", doctag_inst.industry_segment_tag)
        doctag_workflow.add_node("resulting_doc_type", doctag_inst.resulting_doc_type)
        doctag_workflow.add_node("resulting_account_focus", doctag_inst.resulting_account_focus)
        doctag_workflow.add_node("post_processing", doctag_inst.post_processing)

        # Define the flow
        doctag_workflow.add_edge(START, "document_reader")
        doctag_workflow.add_edge("document_reader", "base_product_code")
        doctag_workflow.add_edge("document_reader", "industry_segment")
        doctag_workflow.add_edge("document_reader", "resulting_doc_type")
        doctag_workflow.add_edge("document_reader", "resulting_account_focus")
        doctag_workflow.add_edge("base_product_code", "post_processing")
        doctag_workflow.add_edge("industry_segment", "post_processing")
        doctag_workflow.add_edge("resulting_doc_type", "post_processing")
        doctag_workflow.add_edge("resulting_account_focus", "post_processing")
        doctag_workflow.add_edge("post_processing", END)

        logging.info("End of doctagging_workflow")
        return doctag_workflow.compile()
    
    except Exception as e:
        logging.error(f"Exception occurred in doctagging_workflow: {e}")
        traceback.print_exc()
        return None


