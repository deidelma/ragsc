"""
main.py
"""
from sqlalchemy import create_engine

db_name= 'ragsc'
db_user = 'postgres'
db_pass = 'postgres'
db_host = 'localhost'
db_port = '5432'

db_string = f'postgresql+psycopg://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}'
db = create_engine(db_string)


def test_postgres():
    print("testing postgres")
    print("test completed")

def main():
    print("hello from ragsc")
    test_postgres()


if __name__ == "__main__":
    main()
