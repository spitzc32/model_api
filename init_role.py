from api.db.init_db import SessionLocal
from api.model import Role


def initialize_roles():
    db_session = SessionLocal()
    roles_list = [{"role_name": "Doctor"}, {"role_name": "Researcher"}]
    db_session.bulk_insert_mappings(Role, roles_list)
    db_session.commit()

if __name__ == "__main__":
    initialize_roles()