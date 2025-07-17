from neo4j import GraphDatabase

class EHRKnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def insert_triples(self, ehr_data, patient_id):
        with self.driver.session() as session:
            # Create Patient node
            session.run("""
                MERGE (p:Patient {id: $pid})
            """, pid=patient_id)

            # Diagnoses
            for diag in ehr_data.get("diagnosis", []):
                session.run("""
                    MERGE (p:Patient {id: $pid})
                    MERGE (d:Diagnosis {name: $name})
                    MERGE (p)-[:HAS_DIAGNOSIS]->(d)
                    ON CREATE SET d.first_diagnosed = $date
                    SET d.status = $status,
                        d.description = $description
                """, 
                pid=patient_id,
                name=diag["name"],
                status=diag["status"],
                date=diag["date"],
                description=diag.get("description"))

            # Medications + link to relevant diagnosis if possible
            for med in ehr_data.get("medication", []):
                session.run("""
                    MERGE (p:Patient {id: $pid})
                    MERGE (m:Medication {name: $name})
                    MERGE (p)-[:TAKES]->(m)
                    SET m.dosage = $dosage,
                        m.status = $status,
                        m.indication = $indication,
                        m.start_date = $start_date
                """, 
                pid=patient_id,
                name=med["name"],
                dosage=med["dosage"],
                status=med["status"],
                indication=med["indication"],
                start_date=med.get("start_date"))

                # Optionally, link to Diagnosis by indication match (simple heuristic)
                result = session.run("""
                    MATCH (d:Diagnosis)
                    WHERE d.name CONTAINS $indication OR d.description CONTAINS $indication
                    MATCH (m:Medication {name: $mname})
                    MERGE (m)-[:TREATS]->(d)
                """, 
                indication=med["indication"],
                mname=med["name"])

            # Family History
            for fam in ehr_data.get("family_history", []):
                session.run("""
                    MERGE (p:Patient {id: $pid})
                    MERGE (r:Relative {relation: $relation, patient_id: $pid})
                    MERGE (c:Condition {name: $condition})
                    MERGE (r)-[:HAD_CONDITION]->(c)
                    MERGE (p)-[:HAS_RELATIVE]->(r)
                    SET r.description = $description
                """,
                pid=patient_id,
                relation=fam["relationship"],
                condition=fam["name"],
                description=fam.get("description", ""))

