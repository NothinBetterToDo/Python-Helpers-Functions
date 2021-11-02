import pyodbc
import pandas as pd 


def create_db_connection(driver_name, server_name, db_name, trusted_conn):
    """
    Establish connection to the database using pyodbc 
    :param str driver_name: Types of SQL    
    :param str server_name: Name of the database
    :param str db_name: Name of the server
    :param str trusted_conn: Yes if trusted
    :raises Error: if errors with any of the params 
    """
    connection = None
    try:
        connection = pyodbc.connect(
            Driver = driver_name,
            Server = server_name,
            Database = db_name,
            Trusted_Connection = trusted_conn
        )
        print(f"SQL Server Database connection successful: '{server_name}'")
    except Error as err:
        print(f"Error: '{err}'")
        
    return connection


def create_db(connection, query):
    """
    Create database 
    :param str connection: use create db connection   
    :param str query: create database 'xxx'
    """
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        print("Database created successfully")
    except Error as err:
        print(f"Error: {err}")
        
        
def execute_query(connection, query):
    """
    Execute SQL query
    :param str connection: use create db connection
    :param str query: write SQL query
    """
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        print("Query successful")
    except Error as err:
        print(f"Error: '{err}'")
                
        
def read_query(connection, query):
    """
    Pull data from existing database to feed to a python data frame 
    :param str connection: use create db connection 
    :param str query: write SQL query
    """
    cursor = connection.cursor()
    result = None
    try: 
        cursor.execute(query)
        result = cursor.fetchall()
        return result
    except Error as err:
        print(f"Error: '{err}'")
        
        