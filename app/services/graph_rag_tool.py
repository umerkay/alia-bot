from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from neo4j import GraphDatabase
from typing import Literal
from app.config import Settings

class EHRKGQuery:
    def __init__(self, uri=None, user=None, password=None):
        settings = Settings()
        self.driver = GraphDatabase.driver(
            uri or settings.NEO4J_URI,
            auth=(user or settings.NEO4J_USER, password or settings.NEO4J_PASSWORD)
        )

    def query_ehr(self, patient_id: str, question: str) -> str:
        with self.driver.session() as session:
            if "diagnoses" in question:
                result = session.run("""
                    MATCH (p:Patient {id: $pid})-[:HAS_DIAGNOSIS]->(d:Diagnosis)
                    RETURN d.name AS diagnosis, d.status AS status, d.first_diagnosed AS first_diagnosed, d.description AS description
                """, pid=patient_id)
                records = result.values()
                if not records:
                    return "No diagnoses found."
                return "\n".join([
                    f"- {r[0]} (status: {r[1]}, since: {r[2]})"
                    for r in records
                ])

            elif "medications" in question or "taking" in question:
                result = session.run("""
                    MATCH (p:Patient {id: $pid})-[:TAKES]->(m:Medication)
                    OPTIONAL MATCH (m)-[:TREATS]->(d:Diagnosis)
                    RETURN m.name AS med, m.dosage AS dosage, m.indication AS indication, m.status AS status, m.start_date AS start_date, collect(DISTINCT d.name) AS treats
                """, pid=patient_id)
                records = result.values()
                if not records:
                    return "No medications found."
                return "\n".join([
                    f"- {r[0]} ({r[1]}, for: {r[2]}, status: {r[3]}, started: {r[4]}, treats: {', '.join(r[5]) if r[5] else 'N/A'})"
                    for r in records
                ])

            elif "family_history" in question:
                result = session.run("""
                    MATCH (p:Patient {id: $pid})-[:HAS_RELATIVE]->(r:Relative)-[:HAD_CONDITION]->(c:Condition)
                    RETURN r.relation AS relative, c.name AS condition, r.description AS description
                """, pid=patient_id)
                records = result.values()
                if not records:
                    return "No family history found."
                return "\n".join([
                    f"- {r[0]} had {r[1]} ({r[2]})" for r in records
                ])

            else:
                return "I don't know how to answer that question from the EHR graph."

def create_ehr_retrieval_tool():
    ehr_query = EHRKGQuery()

    @tool
    def ehr_retriever(
        data_type: Literal["diagnoses", "medications", "family_history"],
        config: RunnableConfig
    ) -> str:
        """Searches patient EHR data. Valid types: diagnoses, medications, family_history."""
        try:
            patient_id = config["configurable"].get("patient_id")
            print(f"Retrieving EHR data for patient {patient_id} with type: {data_type}")
            return ehr_query.query_ehr(patient_id, data_type)
        except Exception as e:
            print(f"Error in EHR retrieval: {e}")
            return "Failed to retrieve EHR data."

    return ehr_retriever
