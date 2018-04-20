import MySQLdb as mysql
import pandas as pd

class DB_Client():

    def __init__(self, db_conf=None):

        db_name = 'paygapp'
        host = '0.0.0.0'
        port = 3306
        user = 'admin'
        password = 'admin'
        
        self.conn = mysql.connect(
            db=db_name,
            host=host,
            port=port,
            user=user,
            passwd=password
        )

    def exec_query(self, query):
        cursor = self.conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        self.conn.commit()
        cursor.close()
        return rows

    def get_dataframe(self, query):
        df = pd.read_sql_query(query, self.conn)
        return df