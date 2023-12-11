import json
import psycopg2
from psycopg2 import Error
import psycopg2.extras
from psycopg2.extras import RealDictCursor
import logging
import config

name = config.db_username()
password = config.db_password()
rds_host = config.db_endpoint()
db_name = config.db_name()
port = config.db_port()


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def set_db():
    try:
        conn = psycopg2.connect(host=rds_host, user=name, password=password, dbname=db_name, connect_timeout=30)
    except (Exception, psycopg2.Error) as e:
        logger.error("There is an error in connecting to the DB with error: {}".format(e))
    return conn


def gen_select_query(table_name, columns): #columns is a list of columns to be selected
    qry = """SELECT {} from """+ table_name
    cols = ''
    for key in columns:
        cols += "{},".format(key)
    return qry.format(cols[:-1])

def gen_update_query(table_name, columns, data):
        qry = """UPDATE """ + table_name + """ SET""";
        update = ''
        for key in columns:
            if key in data:
                if type(data[key]) == list:
                    update += """ {}='{}',""".format(key, json.dumps(data[key])) #list of values is dumped as a string into a single records column using json.dumps()
                else: 
                    update += """ {}='{}',""".format(key, data[key])
        if not len(update):
            return None
        return qry + update[:-1]
        
def gen_insert_query(table_name, data):
    qry = """INSERT INTO """ + table_name + """ ({0}) VALUES ({1})"""
    cols = ''
    values = ''
    for key in data:
        cols += "{},".format(key)
        values += "'{}',".format(data[key])
    return qry.format(cols[:-1], values[:-1])
    
def gen_delete_query(table_name):
    qry = """DELETE FROM """ + table_name;
    return qry


def run_query(query_type, query):
        """Execute postgreSQL query"""
        try:
            conn = set_db()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if query_type is not None and (query_type.upper() in ("INSERT", "UPDATE","DELETE")):
                    logger.debug("Query----> {}".format(query))
                    cur.execute(query)
                    conn.commit()
                    affected = cur.rowcount
                    cur.close()
                    return affected
                elif query_type is not None or query_type.upper() == "SELECT":
                    logger.debug("Query----> {}".format(query))
                    cur.execute(query)
                    result = cur.fetchall()
                    cur.close()
                    return result
                else:
                    return "query_type Not Defined"
        except (Exception, psycopg2.Error) as e:
            logger.error("Error in running the query: {}".format(e))
            return 0
        finally:
            if conn:
                conn.close()
                logger.info('Database connection closed.')

def run_select_query(query):
        """Execute postgreSQL query"""
        try:
            conn = set_db()
            cur = conn.cursor()
            logger.debug("Query----> {}".format(query))
            cur.execute(query)
            result = cur.fetchone()
            cur.close()
            return result
        except (Exception, psycopg2.Error) as e:
            logger.error("Error in running the query: {}".format(e))
            return 0
        finally:
            if conn:
                conn.close()
                logger.info('Database connection closed.')