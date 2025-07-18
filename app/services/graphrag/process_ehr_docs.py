import os
import json
import logging
from typing import Dict, Any, List
from pathlib import Path
from app.services.graphrag.ehr_kg import EHRKnowledgeGraph
from app.config import Settings
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EHRDocumentProcessor:
    """
    Processes EHR documents from a directory structure and inserts them into a Neo4j knowledge graph.
    
    Expected directory structure:
    directory/
        patient_id1/
            ehr.json
        patient_id2/
            ehr.json
        ...
    """
    
    def __init__(self, neo4j_uri: str = None, 
                 neo4j_user: str = None, 
                 neo4j_password: str = None):
        """
        Initialize the EHR document processor.
        
        Args:
            neo4j_uri (str): Neo4j database URI
            neo4j_user (str): Neo4j username
            neo4j_password (str): Neo4j password
        """
        settings = Settings()
        self.kg = EHRKnowledgeGraph(
            neo4j_uri or settings.NEO4J_URI,
            neo4j_user or settings.NEO4J_USER,
            neo4j_password or settings.NEO4J_PASSWORD
        )
        self.processed_patients = []
        self.failed_patients = []
        
    def load_ehr_json(self, file_path: str) -> Dict[str, Any]:
        """
        Load and parse an EHR JSON file.
        
        Args:
            file_path (str): Path to the EHR JSON file
            
        Returns:
            Dict[str, Any]: Parsed EHR data
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the file contains invalid JSON
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                ehr_data = json.load(file)
                logger.info(f"Successfully loaded EHR data from {file_path}")
                return ehr_data
        except FileNotFoundError:
            logger.error(f"EHR file not found: {file_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading {file_path}: {e}")
            raise
    
    def validate_ehr_data(self, ehr_data: Dict[str, Any], patient_id: str) -> bool:
        """
        Validate EHR data structure.
        
        Args:
            ehr_data (Dict[str, Any]): EHR data to validate
            patient_id (str): Patient ID for logging
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        required_fields = ['diagnosis', 'medication', 'family_history']
        
        for field in required_fields:
            if field not in ehr_data:
                logger.warning(f"Missing required field '{field}' in EHR data for patient {patient_id}")
                return False
            
            if not isinstance(ehr_data[field], list):
                logger.warning(f"Field '{field}' should be a list in EHR data for patient {patient_id}")
                return False
        
        return True
    
    def process_single_patient(self, patient_dir: str) -> bool:
        """
        Process EHR data for a single patient.
        
        Args:
            patient_dir (str): Path to patient directory
            
        Returns:
            bool: True if processing was successful, False otherwise
        """
        patient_id = os.path.basename(patient_dir)
        ehr_file_path = os.path.join(patient_dir, 'ehr.json')
        
        try:
            # Check if EHR file exists
            if not os.path.exists(ehr_file_path):
                logger.warning(f"No ehr.json file found for patient {patient_id} in {patient_dir}")
                return False
            
            # Load EHR data
            ehr_data = self.load_ehr_json(ehr_file_path)
            
            # Validate data structure
            if not self.validate_ehr_data(ehr_data, patient_id):
                logger.error(f"Invalid EHR data structure for patient {patient_id}")
                return False
            
            # Insert into knowledge graph
            logger.info(f"Inserting EHR data for patient {patient_id} into knowledge graph...")
            self.kg.insert_triples(ehr_data, patient_id=patient_id)
            
            self.processed_patients.append(patient_id)
            logger.info(f"Successfully processed EHR data for patient {patient_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process patient {patient_id}: {e}")
            self.failed_patients.append(patient_id)
            return False
    
    def process_directory(self, directory_path: str) -> Dict[str, Any]:
        """
        Process all patient EHR files in the specified directory.
        
        Args:
            directory_path (str): Path to directory containing patient subdirectories
            
        Returns:
            Dict[str, Any]: Processing summary with success/failure counts and lists
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if not directory_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")
        
        logger.info(f"Starting to process EHR documents in directory: {directory_path}")
        
        # Get all patient directories
        patient_dirs = [d for d in directory_path.iterdir() 
                       if d.is_dir() and not d.name.startswith('.')]
        
        if not patient_dirs:
            logger.warning(f"No patient directories found in {directory_path}")
            return {
                'total_patients': 0,
                'processed_successfully': 0,
                'failed': 0,
                'processed_patients': [],
                'failed_patients': []
            }
        
        logger.info(f"Found {len(patient_dirs)} patient directories")
        
        # Process each patient directory
        for patient_dir in patient_dirs:
            self.process_single_patient(str(patient_dir))
        
        # Generate summary
        summary = {
            'total_patients': len(patient_dirs),
            'processed_successfully': len(self.processed_patients),
            'failed': len(self.failed_patients),
            'processed_patients': self.processed_patients.copy(),
            'failed_patients': self.failed_patients.copy()
        }
        
        logger.info(f"Processing complete. Successfully processed: {summary['processed_successfully']}, "
                   f"Failed: {summary['failed']}")
        
        return summary
    
    def close(self):
        """Close the knowledge graph connection."""
        self.kg.close()
        logger.info("Closed knowledge graph connection")


def main():
    """
    Main function to process EHR documents.
    Example usage of the EHRDocumentProcessor.
    """
    # Configuration
    settings = Settings()
    SHARED_DOCS_DIR = str(settings.shared_docs_path.resolve())
    
    # Initialize processor (will use settings defaults)
    processor = EHRDocumentProcessor()
    
    try:
        # Process all EHR documents
        summary = processor.process_directory(SHARED_DOCS_DIR)
        
        # Print summary
        print("\n" + "="*50)
        print("EHR DOCUMENT PROCESSING SUMMARY")
        print("="*50)
        print(f"Total patients found: {summary['total_patients']}")
        print(f"Successfully processed: {summary['processed_successfully']}")
        print(f"Failed to process: {summary['failed']}")
        
        if summary['processed_patients']:
            print(f"\nSuccessfully processed patients:")
            for patient in summary['processed_patients']:
                print(f"  - {patient}")
        
        if summary['failed_patients']:
            print(f"\nFailed to process patients:")
            for patient in summary['failed_patients']:
                print(f"  - {patient}")
        
        print("="*50)
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        print(f"Error: {e}")
    
    finally:
        # Always close the connection
        processor.close()


if __name__ == "__main__":
    main()