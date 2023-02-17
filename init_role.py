"""
PROGRAM:      root > init_role.py
PROGRAMMER:   Jayra Gaile Ortiz
VERSION 1:    Sept. 20, 2022 
VERSION 2:    Jan. 12, 2023
PURPOSE:      Initialize roles in the database
ALGORITHM:    Initiating the roles to insert inside the database.  

"""
from api.db.init_db import SessionLocal
from api.model import Role


def initialize_roles():
    """
    Initialize required roles for the API. 
    """
    db_session = SessionLocal()
    roles_list = [
        {"role_name": "Doctor"}, 
        {"role_name": "Researcher"},
        {"role_name": "Nurse Assistant"},
        {"role_name": "Nurse"}, 
        {"role_name": "Secretary"} ]
    db_session.bulk_insert_mappings(Role, roles_list)
    db_session.commit()

if __name__ == "__main__":
    initialize_roles()